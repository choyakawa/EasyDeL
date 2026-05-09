# Current Divergence From `upstream/main`

This document is a compact inventory of the branch's current code-level
differences from `upstream/main`.

It intentionally describes only differences that still exist in the working
tree.

## Changed Files

- `easydel/trainers/training_configurations.py`
- `easydel/infra/elarge/types/training.py`
- `easydel/data/transforms/pack.py`
- `easydel/scripts/finetune/__init__.py`
- `easydel/scripts/finetune/train.py`
- `easydel/utils/parameters_transformation.py`
- `easydel/infra/utils.py`
- `easydel/modules/_base/causal_lm_module.py`
- `easydel/trainers/trainer/_fn.py`
- `easydel/trainers/prompt_transforms.py`
- `easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py`
- `easydel/modules/glm/modeling_glm.py`
- `easydel/operations/kernels/paged_flash_attention.py`
- `scripts/mount_gcsfuse.sh`
- `tests/data/test_pack_assistant_masks.py`
- `tests/trainers/test_trainer_forward_kwargs_safety.py`

## 1. Trainer Memory Tracking Typing

The trainer configuration narrows `track_memory` from `bool | float` to `bool`.

Files:

- `easydel/trainers/training_configurations.py`
- `easydel/infra/elarge/types/training.py`

Behavioral effect:

- float intervals are no longer represented in the public typed config surface
- the dataclass and ELM typed config now agree that `track_memory` is boolean-only

## 2. Local SFT Finetune Entrypoint

The branch adds a standalone SFT training script.

Files:

- `easydel/scripts/finetune/__init__.py`
- `easydel/scripts/finetune/train.py`

The script:

- initializes distributed JAX at import time
- parses `ed.SFTConfig` plus a local `RunTimeConfig`
- loads tokenizer and dataset via `HuggingFaceShardedSource`
- chooses image-text or causal-LM EasyDeL auto model from the HF config
- supports `full`, `lora`, and `lora_embed_head` training modes
- optionally marks embedding and LM head parameters trainable through `nn.LoRAParam`
- runs `ed.SFTTrainer`
- unwraps LoRA after training when needed
- converts the final EasyDeL model to a HF-compatible torch checkpoint
- saves only on process 0 after multi-host synchronization

The export path sets:

- `EASY_SAFE_TRANSFER=1`
- `EASYDEL_CHUNK_BYTES=64 * 1024 * 1024`

## 3. Multi-Host and Chunked Export Conversion

The JAX-to-torch conversion path is customized for large sharded runs.

File:

- `easydel/utils/parameters_transformation.py`

Changes:

- imports `jax.experimental.multihost_utils` as `mhutils`
- adds local-shard fallback logic for non-fully-addressable arrays
- detects TPU via `jax.devices()` / `jax.default_backend()`
- adds chunked CPU host transfer controlled by `EASYDEL_CHUNK_BYTES`
- handles bf16 conversion through float32 before converting back to torch bf16
- adds `TensorConverter.global_array_to_host_numpy(...)`
- gathers addressable shards across processes and assembles the full array on process 0
- uses module `_gather_fns` in `StateDictConverter.easydel_to_torch(...)` when available
- changes model conversion checks to warn on missing keys
- prints converted state-dict keys before strict `load_state_dict(...)`

Operational intent:

- avoid converting only local shard shapes
- avoid large single host transfers
- make TPU / multi-host checkpoint export possible from sharded arrays

## 4. LoRA Wrapping Sharding Metadata

LoRA wrapping of `ParallelLinear` layers is extended with dynamic sharding metadata.

File:

- `easydel/infra/utils.py`

Changes:

- wraps matching `ParallelLinear` modules in `eLoRA` as before, but stores the wrapper first
- attaches a dynamic `craft_sharding(...)` method to each wrapper
- chooses LoRA sharding based on the wrapped base module direction:
  - row-parallel base: `lora_a` row-wise, `lora_b` replicated
  - column-parallel base: `lora_a` replicated, `lora_b` column-wise
  - unknown direction: both replicated
- includes base-module sharding specs under `base_module/...`

Operational intent:

- keep LoRA adapter sharding compatible with EasyDeL parallel linear layouts
- preserve base-module sharding information after wrapping

## 5. Trainer Step Graph State Merge

Trainer step functions explicitly merge auxiliary graph state.

File:

- `easydel/trainers/trainer/_fn.py`

Changes in both training and evaluation steps:

- maps `state.graphother` leaves through `jax.lax.stop_gradient(...)` when they look array-like
- merges with `nn.merge(state.graphdef, tree, tree_other)`
- no longer relies only on `state.merge(tree)`

Operational intent:

- preserve non-parameter graph state during step execution
- prevent gradients through that auxiliary graph state

## 6. GLM Architecture Adjustments

The GLM implementation intentionally differs from upstream.

File:

- `easydel/modules/glm/modeling_glm.py`

MLP changes:

- replaces fused `gate_up_proj` with separate `gate_proj`, `up_proj`, and `down_proj`
- computes SwiGLU as `down_proj(act(gate_proj(x)) * up_proj(x))`
- adds explicit checkpoint names for gate, up, down, and output activations

Attention changes:

- overrides projection builders
- `q_proj`, `k_proj`, and `v_proj` use `use_bias=True`
- `o_proj` uses `use_bias=False`
- rotary embedding creation uses:
  - `base=getattr(config, "rope_theta", 100000000.0)`
  - `is_neox_style=False`

Operational intent:

- match local GLM checkpoint structure and rotary assumptions

## 7. Paged Flash Attention Logits Dtype

The paged flash attention call no longer forces bf16 logits.

File:

- `easydel/operations/kernels/paged_flash_attention.py`

Change:

- `logits_dtype=jnp.bfloat16` is commented out in the paged flash attention invocation

Operational intent:

- let the kernel/default attention path choose logits dtype instead of hard-coding bf16

## 8. Leakage-Safe Packed SFT Masks

Packed SFT behavior diverges from upstream to preserve assistant-only loss masks
and packed-sequence attention isolation.

Files:

- `easydel/data/transforms/pack.py`
- `easydel/modules/_base/causal_lm_module.py`
- `easydel/trainers/prompt_transforms.py`
- `easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py`
- `tests/data/test_pack_assistant_masks.py`
- `tests/trainers/test_trainer_forward_kwargs_safety.py`

Packing changes:

- `PackedSequence` carries `position_ids` and arbitrary token-aligned fields.
- `PackedShardedSource` can preserve aligned fields such as:
  - `attention_mask`
  - `completion_mask`
  - `assistant_masks`
  - `labels`
- existing source padding is stripped before packing by using the incoming
  `attention_mask`
- synthetic EOS separator tokens receive:
  - `attention_mask=1`
  - assistant/completion masks set to `0`
  - label fields set to `-100`
- packed padding receives mask value `0` and label value `-100`
- `segment_ids` now follow the stable-branch packed metadata convention:
  non-padding segments start at `1`, padding is `0`
- `position_ids` reset within each packed segment

SFT preprocessing changes:

- `SFTPreprocessTransform` accepts a `padding` flag.
- `SFTTrainer` disables tokenizer `max_length` padding before packing, so the
  packer receives real sequence lengths instead of fully padded rows.
- SFT packing explicitly preserves `attention_mask`, `completion_mask`,
  `assistant_masks`, and `labels`.
- `packing_strategy='wrapped'` is rejected for SFT because it can cut through
  sequence boundaries and cannot preserve packed attention isolation.
- `packing_strategy='bfd'` maps to the lazy `first_fit` packer.

Model-call changes:

- `BaseCausalLMModule.__call__` accepts `segment_ids`.
- when `segment_ids` are provided and no explicit `mask_info` is passed, the
  model builds `MaskInfo.from_segments(...)` using `segment_ids` and
  `position_ids`
- this lets packed SFT batches produce block-diagonal causal attention instead
  of relying only on a flat padding mask

Test coverage:

- packed assistant/completion masks stay aligned across packed segments
- existing source padding is stripped before packing
- packed `segment_ids` build a block-diagonal `MaskInfo` attention mask that
  blocks cross-example attention and masks padding segment `0`
- packed metadata has stable-style `segment_ids` and per-segment `position_ids`
- SFT packing disables pre-padding
- wrapped packing is rejected for leakage-safe SFT
- causal-LM loss shifts `decoder_loss_weights` with labels, so
  assistant-only masks weight the assistant target tokens after next-token
  shifting

Operational intent:

- keep `assistant_only_loss=True` accurate when `packing=True`
- prevent attention leakage across packed examples
- preserve stable-branch packed sequence semantics in the newer lazy data path

2026-05-09 verification notes for the local finetune path:

- the provided `--packing True --assistant_only_loss True --dataset_text_field
  messages --attn_mechanism blocksparse` flow uses `SFTPreprocessTransform`
  with padding disabled before packing, so the packer sees real sequence
  lengths rather than 4096-token padded rows
- SFT packing preserves `assistant_masks`/`completion_mask` as token-aligned
  fields, gives synthetic EOS separators mask value `0`, and gives packed
  padding mask value `0`
- `_preprocess_batch_input` converts `assistant_masks` to `completion_mask`
  and `decoder_loss_weights`, multiplies by `attention_mask`, and exposes
  `decoder_segment_ids`/`decoder_positions`
- causal-LM loss shifts both labels and `decoder_loss_weights`, so the mask
  remains target-token aligned after next-token shifting
- model forward receives either `segment_ids` directly or a prebuilt
  `MaskInfo.from_segments(...)`; blocksparse attention consumes that
  `MaskInfo` instead of relying on a flat padding-only mask
- focused pytest was attempted in the local Windows workspace but could not
  collect because the active Python environment lacks `jax`; `uv run` also
  could not prepare the environment because `uvloop==0.21.0` does not support
  Windows

## 9. GCS Fuse Script Mode

File:

- `scripts/mount_gcsfuse.sh`

Change:

- file mode differs from upstream: executable bit removed (`100755` to `100644`)
