# Local Divergence From `upstream/main`

This document describes the branch's real code-level divergence from `upstream/main` as it exists now.

It is meant for a maintainer who wants to reconstruct the local behavior starting from upstream without replaying local history.

This guide intentionally focuses on behavioral and structural differences. It does not try to document merge commits, and it ignores low-signal metadata-only differences such as the executable-bit change on `scripts/mount_gcsfuse.sh`.

## 1. Training is controlled by raw example consumption, not only optimizer steps

### Motivation

The local workflow treats "how many original examples were consumed" as the real training budget. That matters once packing is enabled, because one optimizer step no longer maps cleanly to one raw sample.

### Behavioral target

Starting from upstream, reproduce these properties:

- training can stop after a configured number of original, unpacked examples
- that raw-item progress survives checkpoint save/load
- source-backed or streaming datasets do not need to be fully scanned just to estimate total steps
- when the raw-item budget is the only active bound, training can run with an open-ended epoch loop

### Reconstruction steps

#### File: `easydel/trainers/training_configurations.py`

Add:

- `max_training_raw_items: int | None`

Its meaning is:

- maximum number of real training examples to consume
- independent from `max_training_steps`
- allowed to drive training by itself when step count is not fixed in advance

This file also differs in one typing detail:

- `track_memory` is narrowed from `bool | float` to `bool`

#### File: `easydel/infra/elarge/types/training.py`

Mirror the same typing change in the typed config surface:

- `track_memory: NotRequired[bool]`

#### File: `easydel/trainers/base_trainer.py`

Introduce raw-progress persistence and raw-progress-aware planning.

Required pieces:

- define `RAW_PROGRESS_JSON_NAME = "easydel-raw-progress.json"`
- on checkpoint resume:
  - load the sidecar JSON if present
  - restore `consumed_train_raw_items`
- on checkpoint save:
  - write the current `consumed_train_raw_items` sidecar

Trainer state added by the local branch:

- `_raw_train_item_count`
- `_raw_eval_item_count`
- `_train_progress_uses_raw_items`
- `_consumed_train_raw_items`
- `_pending_train_progress_update`
- `_resumed_consumed_train_raw_items`

Helper behavior added by the local branch:

- `_safe_len(obj)`
  - first try `len(obj)`
  - then fall back to `estimated_length` metadata when available
- `_effective_train_epoch_limit()`
  - when `max_training_raw_items` is active and `max_training_steps` is not set, allow the effective epoch limit to become `None` if the configured epoch count is still the default
- `_raw_item_limit_active()`
- `_load_raw_progress_state(...)`
- `_save_raw_progress_state(...)`
- `_estimate_training_steps_from_raw_item_limit(...)`
- `_should_stop_training(current_step)`
  - stop when either `max_training_steps` or `max_training_raw_items` is reached
- `_count_raw_progress_items_in_example(...)`
  - derive raw-item count from `segment_ids`
- `_count_raw_progress_items_in_batch(...)`
  - aggregate the per-example logic across a pre-collation batch

Step planning differs too:

- `_resolve_step_count(...)` uses `_effective_train_epoch_limit()` instead of blindly using `num_train_epochs`
- when `max_training_raw_items` is active and `max_training_steps` is not forced:
  - resolve train steps from the raw-item budget instead of only from dataset length

Source-backed iteration differs:

- `_create_dataloader_from_source(...)` accepts `num_epochs: int | None`
- when `num_epochs is None`, it loops forever over the wrapped source

Checkpoint save behavior differs:

- `_save_state(...)` also writes the raw-progress sidecar

Progress-bar behavior differs:

- `create_progress_bar(...)` accepts `total: int | None`
- `log_metrics(...)` updates train progress by raw-item deltas when raw progress is active

#### File: `easydel/trainers/trainer/trainer.py`

Wire the raw-item bookkeeping into the actual trainer loop.

Required behavior:

- import `itertools`
- progress-bar total is chosen this way:
  - `max_training_raw_items` if configured
  - else `raw_train_item_count * effective_epoch_limit` when both are known
  - else unknown total
- on resume:
  - restore `_consumed_train_raw_items`
  - initialize the progress bar from raw items when raw progress is active
- when there is no fixed effective epoch limit:
  - iterate epochs with `itertools.count(start_epoch)`
- inside `_train_epoch(...)`:
  - count raw items before collation
  - after each successful step, accumulate:
    - `_consumed_train_raw_items`
    - `_pending_train_progress_update`
- stop by calling `_should_stop_training(current_step)`
- feed raw progress into per-step metrics:
  - `raw_items=self._consumed_train_raw_items`
  - `raw_items_limit=getattr(self.arguments, "max_training_raw_items", None)`

#### File: `easydel/data/sources/base.py`

Extend `HuggingFaceShardedSource` with metadata-based size discovery.

Required behavior:

- add `_estimated_length`
- resolve it with `datasets.load_dataset_builder(...)`
- read `builder.info.splits[split].num_examples` when available
- expose it through `estimated_length`

#### File: `easydel/data/sources/hf_wrapper.py`

Extend `HFDatasetShardedSource` similarly for iterable datasets.

Required behavior:

- for iterable datasets, attempt to derive `_length` from attached split metadata
- keep `__len__` behavior unchanged for truly unsized iterables
- expose metadata-derived size separately through `estimated_length`

## 2. SFT masking is converted into explicit labels before packing, and packed loss is segment-aware

### Motivation

The local branch depends on prompt/completion masking still being correct after tokenization, packing, and shifted causal loss computation. That requires more than upstream's preprocessing-only masks.

### Behavioral target

After reproducing the local branch:

- SFT preprocessing should emit ignore-indexed `labels`
- packing should preserve `labels` and segment metadata
- packed causal loss should not score token transitions that cross segment boundaries
- model-side mask preparation should understand packed segment metadata even when only `segment_ids` are present

### Reconstruction steps

#### File: `easydel/trainers/prompt_transforms.py`

`SFTPreprocessTransform` differs in two ways.

First, it accepts:

- `pad_to_max_length: bool = True`

and uses `_padding_mode()` so packing can disable preprocessing-time padding.

Second, when prompt masking is active, it materializes labels explicitly:

- add `_apply_completion_labels(result)`
- use `completion_mask` when present
- otherwise fall back to `assistant_masks`
- combine with `attention_mask` when present
- create `labels` with:
  - token id on kept positions
  - `-100` on ignored or padded positions
- raise if assistant/completion-only loss was requested but no mask was produced

Call that helper:

- after conversational tokenization
- after prompt/completion tokenization

#### File: `easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py`

The local SFT trainer differs in two ways.

Prompt masking behavior:

- initialize `mask_prompt` from `assistant_only_loss`
- if `completion_only_loss` is explicitly set, let it override the final masking choice

Packing behavior:

- pass `pad_to_max_length=not packing` into `SFTPreprocessTransform`

#### File: `easydel/data/transforms/pack.py`

All packers are extended so packed training examples preserve supervised-loss metadata.

Required behavior:

- add `_PADDING_SEGMENT_ID = -1`
- add `_build_decoder_positions(...)`
- add `_normalize_sample(...)`
  - clip to `seq_length`
  - keep labels aligned
  - append EOS only when needed and only when there is space
  - append `-100` to labels for inserted EOS

`PackedSequence` gains:

- `labels`
- `decoder_positions`

`PackedSequence.to_dict()` also emits:

- `labels`
- `decoder_segment_ids`
- `decoder_positions`

`GreedyPacker` differs by:

- tracking `_labels`
- accepting `labels` in `add(...)`
- preserving labels through flush/final flush
- padding `labels` with `-100`
- padding `segment_ids` with `-1`
- emitting per-segment-reset `decoder_positions`

`PoolPacker` differs by:

- forwarding optional `labels`
- estimating fit from normalized sample length instead of assuming a fresh EOS every time

`FirstFitPacker` differs by:

- buffering `(tokens, labels, source_id)`
- normalizing before sort/bin-pack
- preserving labels inside each bin
- padding labels with `-100`
- padding `segment_ids` with `-1`
- emitting `decoder_positions`

`PackedShardedSource` differs by:

- reading `labels = example.get("labels")`
- forwarding labels into the active packer
- validating the synthetic shard name before iteration
- iterating the wrapped source's real shard names instead of forwarding `"packed_shard_0"` downstream

#### File: `easydel/infra/base_module.py`

Packed segment metadata also affects model-side input preparation and loss setup.

Required behavior:

- import `MaskInfo`
- in `prepare_inputs_for_call(...)`:
  - when `segment_ids` is present and `mask_info` is absent, derive `mask_info = MaskInfo.from_segments(...)`

Before calling the loss strategy:

- create `loss_batch = dict(batch)`
- when `segment_ids` exists:
  - synthesize `decoder_segment_ids` if missing
  - synthesize `decoder_positions` if missing
- those synthesized fields use `-1` as padding segment id and reset positions to `0` at segment boundaries

#### File: `easydel/infra/loss_utils.py`

Packed causal loss differs in three important ways.

Loss weighting:

- default loss weights use `decoder_target_tokens != -100` instead of `> 0`

Packed-sequence normalization:

- `_sum_weights_per_segment(...)` treats negative segment ids as padding

Shifted causal loss:

- add `_shifted_segment_continuation_mask(...)`
- in `ForCausalLMLoss(...)`, when packed segment ids are present:
  - only keep next-token targets that stay inside the same segment
- build a `loss_batch` for `fixed_cross_entropy(...)` containing:
  - shifted `decoder_target_tokens`
  - shifted `decoder_loss_weights`
  - shifted `decoder_positions`
  - shifted `decoder_segment_ids`

## 3. The repository includes a local finetune entrypoint

### Motivation

The local branch wants a single operational script for the most common SFT workflow, instead of relying only on reusable trainer primitives.

### Behavioral target

The branch should contain one script that:

- initializes distributed JAX
- parses an `SFTConfig` plus runtime arguments
- loads tokenizer and dataset
- selects the proper EasyDeL auto-model
- supports `full`, `lora`, and `lora_embed_head`
- trains
- merges LoRA when needed
- exports an HF-compatible checkpoint in a multi-host-safe way

### Reconstruction steps

#### File: `easydel/scripts/finetune/train.py`

Create a direct training script with this structure:

- call `jax.distributed.initialize()`
- define a runtime dataclass with fields covering:
  - repo id
  - training mode
  - LoRA rank/pattern
  - dataset name/split/subset/streaming/cache dir
  - processor repo id
  - sharding axis and optional DCN sharding axis
  - attention mechanism
  - gradient checkpointing
  - dtype / param_dtype / attention dtypes
- parse `(ed.SFTConfig, RunTimeConfig)` with `DataClassArgumentParser`
- load tokenizer and backfill `pad_token_id` from `eos_token_id` when missing
- build a `HuggingFaceShardedSource`
- inspect the HF config and choose:
  - `AutoEasyDeLModelForImageTextToText`
  - or `AutoEasyDeLModelForCausalLM`
- initialize the model with local sharding/runtime settings

Training modes:

- `full`: no LoRA wrapping
- `lora`: apply LoRA to matching layers
- `lora_embed_head`: apply LoRA and additionally make embedding / lm_head params trainable via `nn.LoRAParam`

Formatting behavior:

- do not inject a custom formatting function
- let message-style data flow directly into SFT preprocessing so the tokenizer chat template can produce assistant masks

Training/export behavior:

- instantiate `ed.SFTTrainer`
- run `trainer.train()`
- recover final state
- unwrap LoRA if used
- restore `LoRAParam` embedding / lm_head params back to plain `Param` when needed
- export with:
  - `EASY_SAFE_TRANSFER=1`
  - `EASYDEL_CHUNK_BYTES=64 * 1024 * 1024`
- synchronize hosts before and after save
- on process 0:
  - build the HF model on `torch.device("meta")`
  - load converted weights with `assign=True`
  - save with `safe_serialization=True`

## 4. Export and LoRA wrapping are customized for multi-host and sharded local workflows

### Motivation

The local branch prioritizes successful export and sharding stability on large distributed runs over keeping the upstream conversion path minimal.

### Behavioral target

After reproducing the local branch:

- JAX-to-torch conversion should work even when only local shards are directly addressable
- multi-host TPU export should gather complete arrays on process 0
- large host transfers should happen in chunks
- LoRA-wrapped linear layers should preserve EasyDeL-specific calling and sharding conventions

### Reconstruction steps

#### File: `easydel/utils/parameters_transformation.py`

Add:

- `from jax.experimental import multihost_utils as mhutils`

Inside `TensorConverter.jax_to_pytorch(...)`, add:

- `_get_local_array(arr)`
  - prefer fully addressable arrays
  - otherwise fall back to the first addressable shard
- more robust platform detection via `jax.devices()` and `jax.default_backend()`
- `_cpu_chunked_transfer(arr)`
  - flatten to 1D
  - host-copy in chunks sized by `EASYDEL_CHUNK_BYTES`
  - reassemble on CPU
  - convert bf16 safely for torch
- TPU-specific path:
  - gather the full array through `global_array_to_host_numpy(...)`
- CPU/GPU safe-transfer path:
  - prefer the chunked host-copy implementation

Add `global_array_to_host_numpy(...)`:

- gather addressable shards per process
- all-gather them across hosts with `mhutils.process_allgather(...)`
- assemble the full host array on process 0

`StateDictConverter.easydel_to_torch(...)` also differs:

- try per-parameter gather functions from `module._gather_fns` before conversion
- that avoids converting only a local shard shape

`ModelConverter` differs slightly too:

- when checking the converted state dict, warn on missing keys
- emit debug prints of the converted key set before `load_state_dict(...)`

#### File: `easydel/infra/utils.py`

`apply_lora_to_layers(...)` differs like this:

- wrap matching `ParallelLinear` modules with `eLoRA`
- attach a dynamic `craft_sharding(...)` method to each wrapper
- choose LoRA sharding based on the wrapped module's direction:
  - row-parallel base: `lora_a` row-wise, `lora_b` replicated
  - column-parallel base: `lora_a` replicated, `lora_b` column-wise
  - otherwise both replicated
- include flattened base-module sharding specs under `base_module/...`

## 5. Trainer step functions explicitly merge auxiliary graph state

### Motivation

The local branch expects the training and evaluation step functions to merge `graphother` explicitly instead of relying only on `state.merge(tree)`.

### Behavioral target

Trainer step execution should preserve the auxiliary graph state and stop gradients through it.

### Reconstruction steps

#### File: `easydel/trainers/trainer/_fn.py`

In both `training_step(...)` and `evaluation_step(...)`:

- build `tree_other` from `state.graphother`
- wrap tensor-like leaves with `jax.lax.stop_gradient(...)`
- merge the module explicitly with:
  - `nn.merge(state.graphdef, tree, tree_other)`

## 6. The local GLM implementation intentionally diverges from upstream

### Motivation

The local branch expects GLM internals that better match the local checkpoints and training assumptions than the upstream implementation does.

### Behavioral target

The GLM MLP and attention projections should follow the local structure, not upstream's current one.

### Reconstruction steps

#### File: `easydel/modules/glm/modeling_glm.py`

`GlmMLP` differs by:

- replacing `gate_up_proj` with separate:
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- computing SwiGLU as:
  - `down_proj(act(gate_proj(x)) * up_proj(x))`

`GlmAttention` differs by overriding projection builders:

- `q_proj`, `k_proj`, `v_proj` use `use_bias=True`
- `o_proj` uses `use_bias=False`
- rotary embedding creation uses:
  - `base=getattr(config, "rope_theta", 100000000.0)`
  - `is_neox_style=False`

## 7. Local attention kernels do not force `logits_dtype`

### Motivation

The local branch avoids hard-coding `logits_dtype=jnp.bfloat16` in these flash-attention call sites.

### Behavioral target

The kernels should inherit the upstream/default dtype behavior instead of forcing bf16 logits.

### Reconstruction steps

#### File: `easydel/operations/kernels/flash_attention.py`

- remove the forced `logits_dtype=jnp.bfloat16` argument from the flash-attention invocation

#### File: `easydel/operations/kernels/paged_flash_attention.py`

- remove the forced `logits_dtype=jnp.bfloat16` argument from the paged flash-attention invocation

## 8. Summary

If the local divergence has been reproduced correctly, these statements should all be true:

- training may be bounded by raw-example consumption rather than only by optimizer steps
- raw-item progress survives checkpoint save/load
- source-backed and streaming datasets can contribute estimated lengths without pre-scanning
- SFT prompt/completion masking becomes explicit `labels` before packing
- packed training examples carry `labels`, `decoder_segment_ids`, and `decoder_positions`
- packed causal loss never crosses segment boundaries when computing next-token loss
- the repository contains a direct SFT entrypoint script for the local workflow
- JAX-to-torch export is multi-host-safe and chunked for large arrays
- LoRA wrappers preserve EasyDeL calling conventions and sharding metadata
- trainer step functions merge `graphother` explicitly
- the local GLM implementation and flash-attention call sites intentionally differ from upstream
