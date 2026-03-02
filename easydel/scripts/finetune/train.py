from dataclasses import field
import os

import jax
from flax import nnx as nn
from eformer.aparser import DataClassArgumentParser
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from transformers import AutoConfig, AutoTokenizer
from jax.experimental import multihost_utils as mhutils

import easydel as ed
from easydel.data import HuggingFaceShardedSource
from easydel.infra.factory import registry
from easydel.modules import *  # noqa # init
from easydel.utils.parameters_transformation import StateDictConverter

jax.distributed.initialize()

VALID_TRAINING_MODES = ("full", "lora", "lora_embed_head")


@auto_pytree
class RunTimeConfig:
    """
    Configuration class for runtime settings.

    Attributes:
        repo_id (str): The repository ID.
        training_mode (str): Training mode — one of "full", "lora", "lora_embed_head".
        lora_rank (int): LoRA rank. Only used when training_mode involves LoRA.
        lora_pattern (str): Regex pattern selecting layers for LoRA adaptation.
        dataset_name (str): The name of the dataset.
        dataset_split (str): The split of the dataset to use.
        processor_repo_id (str | None): The repository ID for the processor. If None, defaults to repo_id.
        sharding_axis (str): The sharding axis dims, comma-separated.
        sharding_dcn_axis (str | None): DCN sharding axis dims for multi-host.
        attn_mechanism: The attention mechanism to use.
        gradient_checkpointing: The gradient checkpointing strategy.
        param_dtype: The data type for model parameters.
        dtype: The data type for general computation.
        attn_dtype: The data type for attention computation.
        attn_softmax_dtype: The data type for attention softmax computation.
    """

    repo_id: str = field(
        metadata={"help": "The repository ID."},
    )
    training_mode: str = field(
        default="lora",
        metadata={
            "help": (
                "Training mode. One of: 'full' (full fine-tuning), "
                "'lora' (LoRA adapters only), "
                "'lora_embed_head' (LoRA + trainable embedding & lm_head)."
            )
        },
    )
    lora_rank: int = field(
        default=256,
        metadata={"help": "LoRA rank. Only used when training_mode involves LoRA."},
    )
    lora_pattern: str = field(
        default=".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
        metadata={"help": "Regex pattern selecting layers for LoRA adaptation."},
    )
    dataset_name: str = field(
        default="trl-lib/Capybara",
        metadata={"help": "The name of the dataset."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "The split of the dataset to use."},
    )
    dataset_subset: str | None = field(
        default=None,
        metadata={"help": "Dataset subset/configuration name (e.g. 'en', 'python')."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset instead of downloading it fully."},
    )
    dataset_cache_dir: str | None = field(
        default=None,
        metadata={"help": "Local cache directory for the dataset."},
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
    sharding_dcn_axis: str | None = field(
        default=None,
        metadata={
            "help": (
                "DCN sharding axis dims for multi-host. Example: '1, 8, 1, 1, 1'. "
                "If provided, you can avoid using -1 in --sharding_axis and explicitly place cross-host splits."
            )
        },
    )

    def __post_init__(self):
        """Post-initialization to set dependent parameters."""
        if self.processor_repo_id is None:
            self.processor_repo_id = self.repo_id
        if isinstance(self.sharding_axis, str):
            self.sharding_axis = tuple(map(int, self.sharding_axis.split(",")))
        if isinstance(self.sharding_dcn_axis, str):
            self.sharding_dcn_axis = tuple(map(int, self.sharding_dcn_axis.split(",")))
        if self.training_mode not in VALID_TRAINING_MODES:
            raise ValueError(
                f"Invalid training_mode '{self.training_mode}'. "
                f"Must be one of {VALID_TRAINING_MODES}."
            )


parser = DataClassArgumentParser((ed.SFTConfig, RunTimeConfig))
sft_config, runtime_config = parser.parse_args_into_dataclasses()

runtime_config: RunTimeConfig
sft_config: ed.SFTConfig

if jax.process_index() == 0:
    print("Training Arguments\n----------------------")
    print(sft_config)
    print(f"Training Mode: {runtime_config.training_mode}")
    if runtime_config.training_mode != "full":
        print(f"LoRA Rank: {runtime_config.lora_rank}")
        print(f"LoRA Pattern: {runtime_config.lora_pattern}")
    print("----------------------")


def main():
    use_lora = runtime_config.training_mode != "full"
    train_embed_head = runtime_config.training_mode == "lora_embed_head"

    processor = AutoTokenizer.from_pretrained(runtime_config.processor_repo_id)

    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    # Load dataset via HuggingFaceShardedSource
    dataset = HuggingFaceShardedSource(
        dataset_name=runtime_config.dataset_name,
        split=runtime_config.dataset_split,
        subset=runtime_config.dataset_subset,
        streaming=runtime_config.streaming,
        cache_dir=runtime_config.dataset_cache_dir,
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
        sharding_dcn_axis_dims=runtime_config.sharding_dcn_axis,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=sft_config.max_length,
            mask_max_position_embeddings=sft_config.max_length,
            attn_dtype=runtime_config.attn_dtype,
            attn_softmax_dtype=runtime_config.attn_softmax_dtype,
            gradient_checkpointing=runtime_config.gradient_checkpointing,
            attn_mechanism=runtime_config.attn_mechanism,
        ),
        platform=ed.EasyDeLPlatforms.JAX,
        param_dtype=runtime_config.param_dtype,
        dtype=runtime_config.dtype,
        precision=jax.lax.Precision.DEFAULT,
        partition_axis=ed.PartitionAxis(),
    )

    # --- Apply LoRA if needed ---
    if use_lora:
        model = model.apply_lora_to_layers(runtime_config.lora_rank, runtime_config.lora_pattern)

    # --- Promote embed & lm_head to trainable (LoRAParam) for lora_embed_head mode ---
    if train_embed_head:
        try:
            embedding_module = model.get_embedding()
            if hasattr(embedding_module, "embedding") and isinstance(getattr(embedding_module, "embedding"), nn.Param):
                embedding_module.embedding = nn.LoRAParam(embedding_module.embedding.value)
        except Exception:
            ...

        try:
            lm_head_module = model.get_lm_head()
            if hasattr(lm_head_module, "kernel") and isinstance(getattr(lm_head_module, "kernel"), nn.Param):
                lm_head_module.kernel = nn.LoRAParam(lm_head_module.kernel.value)
        except Exception:
            ...

    # --- Build formatting function ---
    # When dataset_text_field points to a messages column, apply chat template;
    # when it points to a plain text column the trainer handles it directly (formatting_func=None).
    sample_iter = dataset.open_shard(dataset.shard_names[0])
    dataset_sample = next(sample_iter)
    text_field = sft_config.dataset_text_field
    sample_value = dataset_sample.get(text_field) if text_field else None

    if sample_value is not None and isinstance(sample_value, list):
        # Message-style data → apply chat template
        formatting_func = lambda x: processor.apply_chat_template(x[text_field], tokenize=False)
    else:
        # Plain text / pretrain data → no formatting needed
        formatting_func = None

    trainer = ed.SFTTrainer(
        model=model,
        arguments=sft_config,
        train_dataset=dataset,
        processing_class=processor,
        formatting_func=formatting_func,
    )

    output = trainer.train()

    # Get final training state
    state = getattr(output, "state", None)
    if state is None:
        state = trainer.model_state

    # --- Post-training: merge LoRA / restore params ---
    if use_lora:
        merged_model = state.model.unwrap_lora_to_layers()
    else:
        merged_model = state.model

    if train_embed_head:
        # Restore embedding & lm_head from LoRAParam back to Param
        try:
            _emb = merged_model.get_embedding()
            if hasattr(_emb, "embedding") and isinstance(getattr(_emb, "embedding"), nn.LoRAParam):
                _emb.embedding = nn.Param(_emb.embedding.value)
        except Exception:
            ...

        try:
            _lmh = merged_model.get_lm_head()
            if hasattr(_lmh, "kernel") and isinstance(getattr(_lmh, "kernel"), nn.LoRAParam):
                _lmh.kernel = nn.Param(_lmh.kernel.value)
        except Exception:
            ...

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
        hf_model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="30GB")

    # 再次同步，确保所有进程完成收集
    mhutils.sync_global_devices("after_hf_save")


if __name__ == "__main__":
    main()
