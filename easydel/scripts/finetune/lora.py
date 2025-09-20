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

"""
python -m easydel.scripts.sft_finetune \
	--repo-id meta-llama/Llama-3.2-1B-Instruct \
	--dataset_split train[:10%] \
	--total-batch-size 16 \
	--dataset_text_field messages
"""
LORA_RANK = 256
LORA_PATTERN = ".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"

from dataclasses import field
import os

import jax
from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from transformers import AutoConfig, AutoTokenizer
from jax.experimental import multihost_utils as mhutils

import easydel as ed
from easydel.infra.factory import registry
from easydel.modules import *  # noqa # init
from easydel.utils.parameters_transformation import StateDictConverter


@auto_pytree
class RunTimeConfig:
    """
    Configuration class for runtime settings.

    Attributes:
        repo_id (str): The repository ID.
        dataset_name (str): The name of the dataset. Defaults to "trl-lib/ultrafeedback_binarized".
        dataset_split (str): The split of the dataset to use. Defaults to "train".
        processor_repo_id (tp.Optional[str]): The repository ID for the processor. If None, defaults to repo_id.
        sharding_axis (Tuple[int]): The sharding axis. Defaults to (1, -1, 1, 1, 1).
        attn_mechanism (ed.AttentionMechanisms): The attention mechanism to use.
            Defaults to ed.AttentionMechanisms.VANILLA.
        gradient_checkpointing (ed.EasyDeLGradientCheckPointers): The gradient checkpointing strategy.
            Defaults to ed.EasyDeLGradientCheckPointers.NONE.
        param_dtype (jnp.dtype): The data type for model parameters. Defaults to jnp.bfloat16.
        dtype (jnp.dtype): The data type for general computation. Defaults to jnp.bfloat16.
        attn_dtype (jnp.dtype): The data type for attention computation. Defaults to jnp.bfloat16.
        attn_softmax_dtype (jnp.dtype): The data type for attention softmax computation. Defaults to jnp.float32.
    """

    repo_id: str = field(
        metadata={"help": "The repository ID."},
    )
    dataset_name: str = field(
        default="trl-lib/Capybara",
        metadata={"help": "The name of the dataset."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "The split of the dataset to use."},
    )
    processor_repo_id: str | None = field(
        default=None,
        metadata={"help": "The repository ID for the processor. If None, defaults to repo_id."},
    )
    sharding_axis: str = field(
        default="1, -1, 1, 1, 1",
        metadata={"help": "The sharding axis."},
    )
    attn_mechanism: ed.AttentionMechanisms = field(
        default=ed.AttentionMechanisms.VANILLA,
        metadata={"help": "The attention mechanism to use."},
    )
    gradient_checkpointing: ed.EasyDeLGradientCheckPointers = field(
        default=ed.EasyDeLGradientCheckPointers.NONE,
        metadata={"help": "The gradient checkpointing strategy."},
    )
    param_dtype: jnp.dtype = field(
        default=jnp.bfloat16,
        metadata={"help": "The data type for model parameters."},
    )
    dtype: jnp.dtype = field(
        default=jnp.bfloat16,
        metadata={"help": "The data type for general computation."},
    )
    attn_dtype: jnp.dtype = field(
        default=jnp.bfloat16,
        metadata={"help": "The data type for attention computation."},
    )
    attn_softmax_dtype: jnp.dtype = field(
        default=jnp.float32,
        metadata={"help": "The data type for attention softmax computation."},
    )

    def __post_init__(self):
        """Post-initialization to set dependent parameters."""
        if self.processor_repo_id is None:
            self.processor_repo_id = self.repo_id
        if isinstance(self.sharding_axis, str):
            self.sharding_axis = tuple(map(int, self.sharding_axis.split(",")))


parser = DataClassArgumentParser((ed.SFTConfig, RunTimeConfig))
sft_config, runtime_config = parser.parse_args_into_dataclasses()

runtime_config: RunTimeConfig
sft_config: ed.SFTConfig

if jax.process_index() == 0:
    print("Training Arguments\n----------------------")
    print(sft_config)
    print("----------------------")


def main():
    processor = AutoTokenizer.from_pretrained(runtime_config.processor_repo_id)

    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    # Load dataset
    dataset = load_dataset(
        runtime_config.dataset_name,
        split=runtime_config.dataset_split,
    )

    hf_config = AutoConfig.from_pretrained(runtime_config.repo_id)

    avails = [v.module.__name__ for v in registry.task_registry[ed.TaskType.IMAGE_TEXT_TO_TEXT].values()]

    if hf_config.architectures and any(arch in avails for arch in hf_config.architectures):
        load_module = ed.AutoEasyDeLModelForImageTextToText
    else:
        load_module = ed.AutoEasyDeLModelForCausalLM

    # Initialize model
    model = load_module.from_pretrained(
        runtime_config.repo_id,
        auto_shard_model=True,
        sharding_axis_dims=runtime_config.sharding_axis,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=sft_config.max_sequence_length,
            mask_max_position_embeddings=sft_config.max_sequence_length,
            attn_dtype=runtime_config.attn_dtype,
            attn_softmax_dtype=runtime_config.attn_softmax_dtype,
            gradient_checkpointing=runtime_config.gradient_checkpointing,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=runtime_config.attn_mechanism,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        platform=ed.EasyDeLPlatforms.JAX,
        param_dtype=runtime_config.param_dtype,
        dtype=runtime_config.dtype,
        precision=jax.lax.Precision.DEFAULT,
        partition_axis=ed.PartitionAxis(),
    )

    model = model.apply_lora_to_layers(LORA_RANK, LORA_PATTERN)

    trainer = ed.SFTTrainer(
        model=model,
        arguments=sft_config,
        train_dataset=dataset,
        processing_class=processor,
        formatting_func=lambda x: processor.apply_chat_template(x[sft_config.dataset_text_field], tokenize=False),
    )

    output = trainer.train()

    # Get final training state
    state = getattr(output, "state", None)
    if state is None:
        state = trainer.model_state

    # Merge LoRA adapters into base weights
    merged_model = state.model.unwrap_lora_to_layers()
    
    # 在 TPU 多机环境中，避免先 gather 整模导致 OOM。
    # 让所有进程并行逐参数 allgather，只有 rank0 构建并保存 HF 模型。
    # 限制单次主机拷贝块大小，降低峰值内存占用。
    os.environ.setdefault("EASY_SAFE_TRANSFER", "1")
    os.environ.setdefault("EASYDEL_CHUNK_BYTES", str(64 * 1024 * 1024))  # 64MB

    # 同步后开始转换，每个进程都会参与 allgather 协议
    mhutils.sync_global_devices("before_hf_save")

    # 所有进程都执行逐参数转换（非主进程仅参与 allgather，不返回完整权重）
    state_dict = StateDictConverter.easydel_to_torch(module=merged_model, dtype=runtime_config.param_dtype)

    if jax.process_index() == 0:
        import torch

        save_dir = str(trainer.arguments.get_path())
        base_hf_cls = merged_model.get_torch_loader()._model_mapping[type(merged_model.config)]
        base_config = base_hf_cls.config_class.from_dict(merged_model.config.to_dict())
        # 在 meta 设备上构建，再按键加载，避免显存/内存峰值
        with torch.device("meta"):
            hf_model = base_hf_cls(config=base_config)
            hf_model.load_state_dict(state_dict, assign=True, strict=True)
        hf_model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="20GB")

    # 再次同步，确保所有进程完成收集
    mhutils.sync_global_devices("after_hf_save")


if __name__ == "__main__":
    main()
