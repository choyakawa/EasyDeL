# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import contextlib
import functools
import gc
import os
import typing as tp
import warnings

import jax
import jax.extend
import numpy as np
from jax import dlpack
from jax import numpy as jnp
from jax.experimental import multihost_utils as mhutils
from tqdm.autonotebook import tqdm

from easydel.utils.helpers import check_bool_flag, get_logger

from .analyze_memory import SMPMemoryMonitor
from .traversals import flatten_dict, unflatten_dict

if tp.TYPE_CHECKING:
    from transformers import PreTrainedModel

    from easydel.infra.base_config import EasyDeLBaseConfig
    from easydel.infra.base_module import EasyDeLBaseModule


mem_ops = SMPMemoryMonitor(5)
logger = get_logger(__name__)
EASYDEL_PERFRED_HOST_COPY_INDEX = int(os.getenv("EASYDEL_PERFRED_HOST_COPY_INDEX", "0"))
EASYDEL_PERFRED_HOST_COPY = str(os.getenv("EASYDEL_PERFRED_HOST_COPY", "cpu")).lower()
EASYDEL_PERFRED_HOST_COPY = None if EASYDEL_PERFRED_HOST_COPY == "none" else EASYDEL_PERFRED_HOST_COPY


class DtypeHandler:
    """Handles dtype conversions and operations."""

    @staticmethod
    def get_dtype(dtype: str | jnp.dtype) -> jnp.dtype:
        """Convert string dtype representation to JAX dtype."""
        if isinstance(dtype, str):
            dtype_map = {
                "bf16": jnp.bfloat16,
                "bfloat16": jnp.bfloat16,
                "fp16": jnp.float16,
                "float16": jnp.float16,
                "fp32": jnp.float32,
                "float32": jnp.float32,
                "fp64": jnp.float64,
                "float64": jnp.float64,
                "fp8": jnp.float8_e5m2,
                "fp8_e4m3fn": jnp.float8_e4m3fn,
                "fp8_e4m3fnuz": jnp.float8_e4m3fnuz,
                "fp8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
                "fp8_e5m2": jnp.float8_e5m2,
                "fp8_e5m2fnuz": jnp.float8_e5m2fnuz,
                "float8_e4m3fn": jnp.float8_e4m3fn,
                "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
                "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
                "float8_e5m2": jnp.float8_e5m2,
                "float8_e5m2fnuz": jnp.float8_e5m2fnuz,
            }
            dtype = dtype_map[dtype]
        return dtype

    @staticmethod
    def float_tensor_to_dtype(tensor: tp.Any, dtype: str | jnp.dtype | None) -> tp.Any:
        """Convert float tensor to specified dtype."""
        if dtype is None or dtype == "":
            return tensor

        dtype = DtypeHandler.get_dtype(dtype)
        float_dtypes = (
            jnp.bfloat16,
            jnp.float16,
            jnp.float32,
            jnp.float64,
            jnp.float8_e4m3fn,
            jnp.float8_e4m3fnuz,
            jnp.float8_e4m3b11fnuz,
            jnp.float8_e5m2,
            jnp.float8_e5m2fnuz,
        )

        if getattr(tensor, "dtype", None) in float_dtypes:
            tensor = tensor.astype(dtype)
        return tensor


class TensorConverter:
    """Handles tensor conversions between PyTorch and JAX."""

    @staticmethod
    def convert_pytorch_to_jax(tensor: tp.Any, dtype: jnp.dtype) -> jnp.ndarray:
        """Convert PyTorch tensor to JAX array."""
        if "bfloat16" in str(tensor.dtype):
            tensor = tensor.float()
        return jnp.asarray(tensor.cpu().detach().numpy(), dtype=dtype)

    @staticmethod
    @functools.lru_cache
    def get_torch():
        """Import and return torch module (cached)."""
        import torch

        return torch

    @staticmethod
    def jax_to_pytorch(x: jax.Array) -> tp.Any:
        """Convert JAX array to PyTorch tensor."""
        def _get_local_array(arr):
            """Return an addressable local device array for multi-host safe transfer."""
            try:
                if getattr(arr, "is_fully_addressable", False):
                    return arr
            except Exception:
                ...
            try:
                shards = getattr(arr, "addressable_shards", None)
                if shards and len(shards) > 0:
                    return shards[0].data
            except Exception:
                ...
            return arr

        # 更稳健的平台检测：优先使用已存在设备的平台集合
        try:
            device_platforms = {d.platform for d in jax.devices()}
        except Exception:
            device_platforms = set()
        try:
            default_backend = jax.default_backend()
        except Exception:
            default_backend = None

        def _cpu_chunked_transfer(arr: jax.Array):
            """Safely transfer possibly-large JAX array to torch via CPU in chunks to avoid HBM spikes.

            Strategy:
            - Always flatten to 1D on-device (reshape is cheap/view) to enable uniform chunking.
            - Copy chunks of size <= EASYDEL_CHUNK_BYTES from device to host and stitch on CPU.
            - Finally, reshape back to original shape on CPU and convert to torch.
            """
            import math
            torch = TensorConverter.get_torch()
            local = _get_local_array(arr)
            shape = tuple(local.shape)
            # Fast path for scalars/small tensors
            total_elems = int(np.prod(shape)) if len(shape) > 0 else 1
            dtype_np = np.dtype(str(local.dtype)) if isinstance(local.dtype, jnp.dtype) else local.dtype
            if total_elems == 0:
                return torch.from_numpy(np.array(jax.device_get(local)))

            # Flatten to 1D for consistent chunking
            local_1d = jnp.reshape(local, (total_elems,))
            # Default 128MB per chunk on TPU unless overridden
            chunk_bytes = int(os.getenv("EASYDEL_CHUNK_BYTES", str(128 * 1024 * 1024)))
            bytes_per_elem = np.dtype(dtype_np).itemsize
            elems_per_chunk = max(1, chunk_bytes // max(1, bytes_per_elem))

            host_np_flat = np.empty((total_elems,), dtype=dtype_np)
            for start in range(0, total_elems, elems_per_chunk):
                end = min(total_elems, start + elems_per_chunk)
                # Device to host for a flat slice to minimize transient allocations
                chunk = jax.device_get(local_1d[start:end])
                host_np_flat[start:end] = np.asarray(chunk, dtype=dtype_np)
            host_np = host_np_flat.reshape(shape)
            # torch.from_numpy 不支持 ml_dtypes.bfloat16，需转换
            if str(dtype_np) in ("bfloat16", "bf16"):
                return torch.from_numpy(host_np.astype(np.float32)).to(torch.bfloat16)
            return torch.from_numpy(host_np)

        # TPU 专用路径：使用多主机安全的全局收集，避免一次性在某个 TPU 上分配大缓冲
        if ("tpu" in device_platforms) or (isinstance(default_backend, str) and default_backend.lower() == "tpu"):
            host_np = TensorConverter.global_array_to_host_numpy(x, x.dtype)
            torch = TensorConverter.get_torch()
            if host_np is None:
                # 非主进程不返回内容；这里返回一个占位 0 张量，调用方在主进程执行
                return torch.zeros((), dtype=torch.float32)
            if str(host_np.dtype) in ("bfloat16", "bf16"):
                return torch.from_numpy(host_np.astype(np.float32)).to(torch.bfloat16)
            return torch.from_numpy(host_np)

        # CPU/GPU 平台或强制安全转移：走分块 CPU 转移，避免单次大拷贝
        if check_bool_flag("EASY_SAFE_TRANSFER", True):
            try:
                return _cpu_chunked_transfer(x)
            except Exception:
                return _cpu_chunked_transfer(_get_local_array(x))
        else:
            from torch import cuda
            from torch.utils import dlpack as dlpack_pt

            cpu_force = not cuda.is_available()

            def _to_dlpack_capsule(arr):
                """Create a DLPack capsule from a JAX array, compatible with JAX >= 0.7."""
                try:
                    # Preferred path on modern JAX: use __dlpack__ on a local-addressable view
                    _arr = _get_local_array(arr)
                    return _arr.__dlpack__()
                except Exception:
                    # Fallback for older JAX versions
                    if hasattr(dlpack, "to_dlpack"):
                        try:
                            return dlpack.to_dlpack(_get_local_array(arr))
                        except Exception:
                            ...
                    # Last resort: host copy then try again
                    host_arr = jax.device_get(_get_local_array(arr))
                    return host_arr.__dlpack__() if hasattr(host_arr, "__dlpack__") else dlpack.to_dlpack(host_arr)

            if (
                str(platform).lower() in ["cpu", "gpu"]
                and not cpu_force
                and not check_bool_flag("EASYDEL_FORCE_TORCH_USE_CPU", False)
            ):
                capsule = _to_dlpack_capsule(x)
            else:
                y = jax.device_put(
                    jax.device_get(_get_local_array(x)),
                    jax.devices(EASYDEL_PERFRED_HOST_COPY)[EASYDEL_PERFRED_HOST_COPY_INDEX],
                )
                capsule = _to_dlpack_capsule(y)
            return dlpack_pt.from_dlpack(capsule)

    @staticmethod
    def global_array_to_host_numpy(x: jax.Array, target_dtype: jnp.dtype) -> np.ndarray | None:
        """Gather a potentially multi-host sharded JAX array into host numpy on process 0.

        - On fully addressable arrays: simple device_get.
        - On sharded arrays: gather addressable shards per process, all-gather across processes,
          and assemble the full array on process 0 CPU. Other processes return None.
        """
        is_main = jax.process_index() == 0

        # Fast path: already fully addressable
        try:
            if getattr(x, "is_fully_addressable", False):
                host_arr = jax.device_get(x)
                return np.asarray(host_arr, dtype=np.dtype(DtypeHandler.get_dtype(target_dtype)))
        except Exception:
            ...

        # Sharded path: collect local shards
        local_chunks: list[tuple[list[tuple[int, int]], np.ndarray]] = []
        try:
            shards = getattr(x, "addressable_shards", None)
        except Exception:
            shards = None

        if shards is None or len(shards) == 0:
            # Fallback: try to get a local view; may still fail if non-addressable
            try:
                host_arr = jax.device_get(x)
                return np.asarray(host_arr, dtype=np.dtype(DtypeHandler.get_dtype(target_dtype)))
            except Exception:
                # As a last resort, return None on non-main to avoid crashes
                return None if not is_main else np.asarray(
                    jax.device_get(x.addressable_shards[0].data),
                    dtype=np.dtype(DtypeHandler.get_dtype(target_dtype)),
                )

        for shard in shards:
            idx_pairs: list[tuple[int, int]] = []
            for dim, sl in enumerate(shard.index):
                start = 0 if sl.start is None else int(sl.start)
                stop = x.shape[dim] if sl.stop is None else int(sl.stop)
                idx_pairs.append((start, stop))
            chunk_np = np.asarray(
                jax.device_get(shard.data),
                dtype=np.dtype(DtypeHandler.get_dtype(target_dtype)),
            )
            local_chunks.append((idx_pairs, chunk_np))

        # Cross-host gather (host-only). All processes must participate.
        gathered: list[list[tuple[list[tuple[int, int]], np.ndarray]]] = mhutils.process_allgather(local_chunks)

        if not is_main:
            # Other processes do not assemble to save memory
            return None

        # Assemble on process 0
        full_np = np.empty(x.shape, dtype=np.dtype(DtypeHandler.get_dtype(target_dtype)))
        for proc_chunks in gathered:
            for idx_pairs, chunk in proc_chunks:
                # 兼容 (start, stop) 或 (start, stop, step) 或直接 slice 对象
                slices_list = []
                for item in idx_pairs:
                    if isinstance(item, slice):
                        s = item
                    elif isinstance(item, (list, tuple)):
                        if len(item) == 2:
                            s = slice(item[0], item[1])
                        elif len(item) >= 3:
                            s = slice(item[0], item[1], item[2])
                        else:
                            s = slice(0, None)
                    else:
                        # 单值索引不预期，这里退化为范围 1 的切片
                        try:
                            idx_int = int(item)
                            s = slice(idx_int, idx_int + 1)
                        except Exception:
                            s = slice(0, None)
                    slices_list.append(s)
                slices = tuple(slices_list)
                full_np[slices] = chunk
        return full_np

    @staticmethod
    def pytorch_to_jax(x: tp.Any) -> jnp.ndarray:
        """Convert PyTorch tensor to JAX array."""
        return jnp.asarray(x.detach().cpu().numpy())


class StateDictConverter:
    """Handles conversion between PyTorch and EasyDeL state dictionaries."""

    @staticmethod
    def match_keywords(string: str, required: list[str], forbidden: list[str]) -> bool:
        """Check if string contains all required keywords and none of the forbidden ones."""
        return all(t in string for t in required) and not any(n in string for n in forbidden)

    @staticmethod
    def process_tensor(key: str, tensor: tp.Any, config: dict[str, tp.Any]) -> tuple[tuple, jnp.ndarray] | None:
        """Process a single tensor and return its processed key and value."""
        new_key = key

        if any(layer_name in key for layer_name in config["embedding_layer_names"]):
            new_key = f"{key[: -len('.weight')]}.embedding"

        elif any(layer_norm in key for layer_norm in config["layernorm_names"]):
            new_key = key.replace(".weight", ".scale")

        elif "weight" in key:
            is_moe_expert = key in config.get("consolidated_moe_keys", set())
            ndim = len(tensor.shape)
            if not is_moe_expert:
                if ndim == 2:
                    tensor = tensor.permute(1, 0)
                elif ndim == 3:
                    tensor = tensor.permute(2, 1, 0)
                elif ndim == 4:
                    tensor = tensor.permute(2, 3, 1, 0)
                elif ndim == 5:
                    tensor = tensor.permute(2, 3, 4, 1, 0)
                elif ndim == 6:
                    tensor = tensor.permute(4, 5, 3, 2, 1, 0)
            else:
                if ndim == 3:
                    tensor = tensor.permute(0, 2, 1)
            new_key = key.replace(".weight", ".kernel")

        key_tuple = tuple(int(n) if n.isdigit() else n for n in new_key.split("."))

        if config["uses_tie_word_embedding"] and config["lm_head_name"] and key_tuple[0] == config["lm_head_name"]:
            return None

        array = TensorConverter.convert_pytorch_to_jax(tensor, config["dtype"])
        return key_tuple, array

    @staticmethod
    def _base_huggingface_to_easydel(
        state_dict: dict[str, tp.Any],
        *,
        device: jax.Device | None = None,  # type:ignore
        embedding_layer_names: list[str] | None = None,
        layernorm_names: list[str] | None = None,
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        shard_fns: tp.Mapping[tuple, tp.Callable] | None = None,
        dtype: jnp.dtype = jnp.float16,
        verbose: bool = True,
        callback: tp.Callable[[jax.Array, tuple], jax.Array] | None = None,
        remove_state_dict: bool = False,
        lm_head_name: str | None = None,
        uses_tie_word_embedding: bool = False,
        consolidated_moe_keys: set[str] | None = None,
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Base conversion function from PyTorch state dict to EasyDeL format."""
        try:
            import torch

            _clear = torch.cuda.empty_cache if torch.cuda.is_available() else gc.collect
        except ModuleNotFoundError:
            _clear = gc.collect

        config = {
            "embedding_layer_names": set(embedding_layer_names or []),
            "layernorm_names": set(layernorm_names or []),
            "moe_block_names": set(moe_block_names or []),
            "moe_names": set(moe_names or []),
            "lm_head_name": lm_head_name,
            "uses_tie_word_embedding": uses_tie_word_embedding,
            "dtype": dtype,
            "consolidated_moe_keys": consolidated_moe_keys or set(),
        }

        with jax.default_device(device) if device is not None and shard_fns is None else contextlib.nullcontext():
            flax_dict = {}
            with tqdm(total=len(state_dict), disable=not verbose, desc="Converting Model") as pbar:
                for key, tensor in state_dict.items():
                    try:
                        result = StateDictConverter.process_tensor(key, tensor, config)
                        if result is not None:
                            key_tuple, jax_array = result
                            if shard_fns and key_tuple in shard_fns:
                                jax_array = shard_fns[key_tuple](jax_array)
                            if callback is not None:
                                jax_array = callback(jax_array, key_tuple)
                            flax_dict[key_tuple] = jax_array
                    except Exception as e:
                        logger.error(f"Error processing key {key}: {e!s}")
                    pbar.update(1)

            if remove_state_dict:
                del state_dict
                _clear()

            return unflatten_dict(flax_dict)

    @staticmethod
    def apply_moe_transformations(
        state_dict: dict[str, tp.Any],
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        moe_block_path: list[str] | None = None,
        moe_path: list[str] | None = None,
        tensor_transform: tp.Callable | None = None,
    ) -> tuple[dict[str, tp.Any], set[str]]:
        """
        Transform MoE weights from HuggingFace format (separate experts) to EasyDel format (stacked experts).
        Converts from:
            model.layers.3.block_sparse_moe.experts.0.w3.weight -> shape (128, 256)
            model.layers.3.block_sparse_moe.experts.1.w3.weight -> shape (128, 256)
            ...
        To:
            model.layers.3.block_sparse_moe.experts.w3.weight -> shape (num_experts, 128, 256)
        """
        if not all([moe_block_names, moe_names, moe_block_path]):
            return state_dict, set()

        import torch

        excepted_expert_name = moe_path[0].split(".")[-2]
        expert_prefix = f".{excepted_expert_name}."

        moe_names_set = set(moe_names)
        moe_stacked_paths = {
            f"{block_path}.{excepted_expert_name}.{moe_name}" for block_path in moe_block_path for moe_name in moe_names
        }

        new_state_dict = {}
        moe_groups = {path: {} for path in moe_stacked_paths}
        consolidated_moe_keys = set()

        for key in tqdm(list(state_dict.keys()), desc="Applying MoE Transformations"):
            is_moe_expert = False
            value = state_dict.pop(key)
            if expert_prefix not in key:
                new_state_dict[key] = value
                continue

            for block_path in moe_block_path:
                block_expert_prefix = block_path + expert_prefix
                if key.startswith(block_expert_prefix):
                    remainder = key[len(block_expert_prefix) :]

                    dot_idx = remainder.find(".")
                    if dot_idx <= 0:
                        continue

                    expert_part = remainder[:dot_idx]
                    if not expert_part.isdigit():
                        continue

                    expert_idx = int(expert_part)
                    moe_name_part = remainder[dot_idx + 1 :]
                    moe_name = moe_name_part[:-7] if moe_name_part.endswith(".weight") else moe_name_part

                    if moe_name in moe_names_set:
                        target_path = f"{block_path}.{excepted_expert_name}.{moe_name}"
                        moe_groups[target_path][expert_idx] = value
                        is_moe_expert = True
                        break

            if not is_moe_expert:
                new_state_dict[key] = value
        for target_path, expert_dict in moe_groups.items():
            if not expert_dict:
                continue

            expert_indices = sorted(expert_dict.keys())
            num_experts = len(expert_indices)
            first_tensor = expert_dict[expert_indices[0]]
            new_key = f"{target_path}.weight"

            try:
                if isinstance(first_tensor, torch.Tensor):
                    if first_tensor.device.type != "meta":
                        meta_sample = torch.empty_like(first_tensor, device="meta")
                    else:
                        meta_sample = first_tensor
                    stacked_shape = (num_experts, *meta_sample.shape)
                    stacked_tensor = torch.empty(
                        stacked_shape,
                        dtype=first_tensor.dtype,
                        device=first_tensor.device,
                    )

                    for i, idx in enumerate(expert_indices):
                        stacked_tensor[i] = expert_dict[idx]

                else:
                    import numpy as np

                    expert_tensors = [expert_dict[idx] for idx in expert_indices]
                    stacked_tensor = np.stack(expert_tensors, axis=0)

                if tensor_transform is not None:
                    stacked_tensor = tensor_transform(stacked_tensor)

                new_state_dict[new_key] = stacked_tensor
                consolidated_moe_keys.add(new_key)
            except Exception as e:
                logger.error(f"Failed to stack MoE tensors for {target_path}: {e}")
                for idx, tensor in expert_dict.items():
                    fallback_key = (
                        f"{target_path.replace(f'.{excepted_expert_name}.', f'.{excepted_expert_name}.{idx}.')}.weight"
                    )
                    new_state_dict[fallback_key] = tensor

        return new_state_dict, consolidated_moe_keys

    @staticmethod
    def huggingface_to_easydel(
        state_dict: dict[str, tp.Any],
        *,
        device: jax.Device | None = None,  # type:ignore
        embedding_layer_names: list[str] | None = None,
        layernorm_names: list[str] | None = None,
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        moe_block_path: list[str] | None = None,
        moe_path: list[str] | None = None,
        shard_fns: tp.Mapping[tuple, tp.Callable] | None = None,
        dtype: jnp.dtype = jnp.float16,
        verbose: bool = True,
        callback: tp.Callable[[jax.Array, tuple], jax.Array] | None = None,
        remove_state_dict: bool = False,
        lm_head_name: str | None = None,
        uses_tie_word_embedding: bool = False,
        **kwargs,
    ) -> dict[str, tp.Any]:
        """Convert PyTorch state dict to EasyDeL format with MoE transformations."""
        consolidated_moe_keys = set()
        if moe_block_names is not None and moe_names is not None:
            state_dict, consolidated_moe_keys = StateDictConverter.apply_moe_transformations(
                state_dict=state_dict,
                moe_names=moe_names,
                moe_path=moe_path,
                moe_block_names=moe_block_names,
                moe_block_path=moe_block_path,
            )

        return StateDictConverter._base_huggingface_to_easydel(
            state_dict,
            device=device,
            embedding_layer_names=embedding_layer_names,
            layernorm_names=layernorm_names,
            moe_names=moe_names,
            moe_path=moe_path,
            moe_block_names=moe_block_names,
            moe_block_path=moe_block_path,
            shard_fns=shard_fns,
            dtype=dtype,
            verbose=verbose,
            callback=callback,
            remove_state_dict=remove_state_dict,
            lm_head_name=lm_head_name,
            uses_tie_word_embedding=uses_tie_word_embedding,
            consolidated_moe_keys=consolidated_moe_keys,
            **kwargs,
        )

    @staticmethod
    def apply_moe_transformations_reverse(
        state_dict: dict[str, tp.Any],
        moe_block_names: list[str] | None = None,
        moe_names: list[str] | None = None,
        moe_block_path: list[str] | None = None,
        moe_path: list[str] | None = None,
        tensor_transform: tp.Callable | None = None,
    ) -> dict[str, tp.Any]:
        """
        Transform MoE weights from EasyDel format (stacked experts) to HuggingFace format (separate experts).

        Converts from:
            model.layers.3.block_sparse_moe.experts.w3.weight -> shape (num_experts, 128, 256)
        To:
            model.layers.3.block_sparse_moe.experts.0.w3.weight -> shape (128, 256)
            model.layers.3.block_sparse_moe.experts.1.w3.weight -> shape (128, 256)
            ...
        """
        if not all([moe_block_names, moe_names, moe_block_path]):
            return state_dict

        new_state_dict = {}
        processed_keys = set()
        excepted_expert_name = moe_path[0].split(".")[-2] if moe_path else "experts"

        for key, value in state_dict.items():
            is_stacked_moe = False
            for block_path in moe_block_path:
                if key.startswith(block_path):
                    remainder = key[len(block_path) + 1 :]
                    parts = remainder.split(".")
                    if (
                        len(parts) == 3
                        and parts[0] == excepted_expert_name
                        and parts[1] in moe_names
                        and parts[2] == "weight"
                    ):
                        is_stacked_moe = True
                        moe_name = parts[1]
                        if hasattr(value, "shape") and len(value.shape) >= 3:
                            num_experts = value.shape[0]

                            for expert_idx in range(num_experts):
                                expert_tensor = value[expert_idx]
                                if tensor_transform is not None:
                                    expert_tensor = tensor_transform(expert_tensor)
                                new_key = f"{block_path}.{excepted_expert_name}.{expert_idx}.{moe_name}.weight"
                                new_state_dict[new_key] = expert_tensor

                            processed_keys.add(key)
                            break

            if not is_stacked_moe:
                new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def easydel_to_torch(module: EasyDeLBaseModule, dtype: jnp.dtype = jnp.float16) -> dict[str, tp.Any]:
        """Convert EasyDeL module to PyTorch state dict.

        为确保多机/多设备分片权重被正确合并，同时避免一次性全模型 gather 造成 OOM，
        在逐参数级别上优先尝试使用模块自带的 gather 函数将单个参数聚合为全局张量，
        随后立刻进行主机分块拷贝到 torch，从而控制峰值内存占用。
        """
        if dtype is None:
            dtype = module.param_dtype

        graphtree = unflatten_dict(module.parameters)
        model_parameters = flatten_dict(graphtree, sep=".")


        from easydel.layers.moe import BaseMoeModule, ParallelMoELinear
        from easydel.utils import traversals

        md = ParallelMoELinear
        moe_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(module, md)]
        md = BaseMoeModule
        moe_block_path = [".".join(tuple(map(str, pa))) for pa, _ in traversals.iter_module_search(module, md)]

        moe_names = list(set([names.split(".")[-1] for names in moe_path])) if moe_path else None
        moe_block_names = list(set([names.split(".")[-1] for names in moe_block_path])) if moe_block_path else None

        stacked_moe_keys = set()
        if moe_block_names and moe_names and moe_block_path:
            for block_path in moe_block_path:
                for moe_name in moe_names:
                    potential_key = f"{block_path}.experts.{moe_name}.kernel"
                    if potential_key in model_parameters:
                        stacked_moe_keys.add(potential_key)
        torch_state_dict = {}
        with tqdm(model_parameters.items(), desc=f"Converting {module.__class__.__name__} to torch") as pbar:
            for key, tensor in pbar:
                if tensor is None:
                    continue
                if hasattr(tensor, "materialize"):
                    tensor = tensor.materialize()
                if hasattr(tensor, "value") and hasattr(tensor.value, "materialize"):
                    tensor = tensor.value.materialize()
                tensor = TensorConverter.jax_to_pytorch(jax.block_until_ready(tensor))
                is_stacked_moe = key in stacked_moe_keys

                if key.endswith(".kernel"):
                    if not is_stacked_moe:
                        if tensor.ndim == 2:
                            tensor = tensor.permute(1, 0)
                        elif tensor.ndim == 3:
                            tensor = tensor.permute(2, 1, 0)
                        elif tensor.ndim == 4:
                            tensor = tensor.permute(3, 2, 0, 1)
                        elif tensor.ndim == 5:
                            tensor = tensor.permute(4, 3, 0, 1, 2)
                        elif tensor.ndim == 6:
                            tensor = tensor.permute(5, 4, 3, 2, 0, 1)
                    else:
                        if tensor.ndim == 3:
                            tensor = tensor.permute(0, 2, 1)

                key = key.replace(".kernel", ".weight").replace(".embedding", ".weight").replace(".scale", ".weight")
                torch_state_dict[key] = tensor

        if moe_block_names and moe_names and moe_block_path and moe_path:
            torch_state_dict = StateDictConverter.apply_moe_transformations_reverse(
                state_dict=torch_state_dict,
                moe_names=moe_names,
                moe_path=moe_path,
                moe_block_names=moe_block_names,
                moe_block_path=moe_block_path,
            )

        return torch_state_dict


class ModelConverter:
    """Handles model conversions between EasyDeL and HuggingFace formats."""

    @staticmethod
    def easydel_to_huggingface(
        module: EasyDeLBaseModule,
        config: EasyDeLBaseConfig,
        base_huggingface_module: PreTrainedModel,
        base_huggingface_module_kwarguments: dict | None = None,
        dtype: jnp.dtype = jnp.float16,
        use_meta_torch: bool = True,
        **kw,
    ) -> tp.Any:
        """Convert EasyDeL module to HuggingFace model."""

        import torch

        if base_huggingface_module_kwarguments is None:
            base_huggingface_module_kwarguments = {}

        state_dict = StateDictConverter.easydel_to_torch(module=module, dtype=dtype)
        base_config = base_huggingface_module.config_class.from_dict(config.to_dict())
        with torch.device("meta") if use_meta_torch else contextlib.nullcontext():
            model: torch.nn.Module = base_huggingface_module(config=base_config, **base_huggingface_module_kwarguments)
            key_shape_checks = {k: v.shape for k, v in model.state_dict().items() if hasattr(v, "shape")}
            if len(list(key_shape_checks.keys())) != len(list(state_dict.keys())):
                warnings.warn("There might be an issue with converted `state_dict`.", stacklevel=1)
            for key, shape in key_shape_checks.items():
                if key not in state_dict:
                    warnings.warn(f"Missing {key}.", stacklevel=1)
                elif state_dict[key].shape != shape:
                    warnings.warn(f"Shape conflict at {key}.", stacklevel=1)
            print("state_dict:")
            for key in state_dict:
                print(key)
            model.load_state_dict(state_dict, assign=True, strict=True)

        return model
