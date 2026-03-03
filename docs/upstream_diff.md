# Local Divergence From Upstream: Reconstruction Guide

This document explains the current branch's intentional divergence from upstream in a form that is meant to be actionable.

It is written for a maintainer who starts from upstream and wants to recreate the local behavior without reading local commit history.

The structure is:

1. Motivation: why this divergence exists.
2. Behavioral target: what should be true after the change.
3. Reconstruction steps: which files differ and what to change in each file.

This document focuses on the actual code differences that exist today.

## 1. Training must be controllable by raw sample consumption, not only by optimizer steps

### Motivation

Upstream training control is step-centric. That works when one training example corresponds to one raw example, but it becomes inaccurate once sequence packing is enabled. In the local workflow, the real budget is "how many original examples were consumed", not "how many optimizer updates happened".

The local branch therefore treats raw-item consumption as a first-class training control signal.

### Behavioral target

After reproducing this divergence, the trainer should be able to:

- stop after consuming a configured number of original training examples, even when packing is enabled
- resume that raw-item progress from checkpoints
- show progress in terms of raw items when the dataset format makes that possible
- continue to work with streaming / iterable Hugging Face datasets by using metadata-derived length estimates when exact length is unavailable

### Reconstruction steps

#### File: `easydel/trainers/training_configurations.py`

Add a new training argument:

- `max_training_raw_items: int | None`

Its meaning should be:

- maximum number of real, unpacked training examples to consume before stopping
- independent from `max_training_steps`
- allowed to drive training even when the number of optimizer steps is not fixed upfront

This file also differs in one additional way:

- `track_memory` is narrowed from `bool | float` to `bool`

That narrowing is part of the local divergence and should be reproduced if the goal is an exact mirror of the current branch.

#### File: `easydel/trainers/base_trainer.py`

Introduce raw-progress tracking and raw-progress persistence.

Required behavior:

- define a sidecar filename for checkpointed raw progress:
  - `RAW_PROGRESS_JSON_NAME = "easydel-raw-progress.json"`
- when resuming from checkpoint:
  - load that sidecar if present
  - restore `consumed_train_raw_items`
- when saving checkpoints:
  - write the current `consumed_train_raw_items` to that sidecar

Add trainer state needed for raw progress:

- `_raw_train_item_count`
- `_raw_eval_item_count`
- `_train_progress_uses_raw_items`
- `_consumed_train_raw_items`
- `_pending_train_progress_update`
- `_resumed_consumed_train_raw_items`

Add helper methods for length and progress bookkeeping:

- `_safe_len(obj) -> int | None`
  - first try `len(obj)`
  - if that fails, read `estimated_length` when available
- `_load_raw_progress_state(checkpoint_dir)`
- `_save_raw_progress_state(checkpoint_dir)`
- `_raw_item_limit_active()`
- `_should_stop_training(current_step)`
  - stop when either `max_training_steps` or `max_training_raw_items` is reached
- `_count_raw_progress_items_in_example(example)`
  - use `segment_ids` to count how many original samples are represented by one packed example
  - if `segment_ids` is missing, fall back to `1`
- `_count_raw_progress_items_in_batch(batch)`
  - aggregate the example-level logic above across a pre-collation batch

Add step-resolution helpers that do not pre-scan streaming data:

- `_resolve_configured_step_fallback(is_train)`
- `_calculate_steps_without_scanning(dataset, is_train=...)`
  - use `len()` or `estimated_length` when available
  - otherwise require `per_epoch_training_steps` / `per_epoch_evaluation_steps`
- `_estimate_training_steps_from_raw_item_limit(raw_item_limit, effective_train_epochs)`
  - derive an estimate from raw-item budget instead of pre-iterating the dataset

Change dataloader configuration behavior:

- do not fully pre-iterate source-backed datasets to compute step counts
- for train/eval step resolution:
  - prefer forced steps if configured
  - otherwise prefer exact length
  - otherwise prefer metadata-derived estimated length
  - otherwise require explicit per-epoch step configuration
- when `max_training_raw_items` is active, estimate total training steps from that budget instead of consuming the source once just to count

Change source-backed iteration behavior:

- `_create_dataloader_from_source(...)` must accept `num_epochs: int | None`
- if `num_epochs is None`, iterate indefinitely over the source
- this supports the case where training should continue until raw-item budget is reached rather than until a predetermined number of epochs is exhausted

Change checkpoint save behavior:

- after `state.save_state(...)`, also save the raw-progress sidecar

Change progress-bar update behavior:

- in `log_metrics(...)`, if train progress is raw-item based, update the progress bar using `_pending_train_progress_update`
- otherwise keep the old step-based update behavior

#### File: `easydel/trainers/trainer/trainer.py`

Wire the raw-item logic into the actual training loop.

Required behavior:

- when creating the train progress bar:
  - if `max_training_raw_items` is set, use that as the progress total
  - else if raw item count is known and epoch count is known, use `raw_train_item_count * epochs`
  - else allow progress total to remain unknown
- when resuming:
  - restore `_consumed_train_raw_items`
  - if raw progress is active, initialize the progress bar from raw-item count instead of from step count
- during training:
  - call `_count_raw_progress_items_in_batch(batch)` before collation
  - after a successful train step, accumulate:
    - `_consumed_train_raw_items`
    - `_pending_train_progress_update`
- stop conditions:
  - call `_should_stop_training(current_step)` instead of checking only `current_step >= max_training_steps`
- pass raw progress into step metrics:
  - `raw_items=self._consumed_train_raw_items`
  - `raw_items_limit=getattr(self.arguments, "max_training_raw_items", None)`

Also change epoch iteration behavior:

- when there is no effective fixed epoch limit, iterate with `itertools.count(start_epoch)` instead of `range(...)`

#### File: `easydel/data/sources/base.py`

Extend `HuggingFaceShardedSource` with metadata-based length estimation.

Required behavior:

- add `_estimated_length`
- populate it using `datasets.load_dataset_builder(...)`
- look up `builder.info.splits[split].num_examples`
- expose it via an `estimated_length` property

This is needed so raw-progress and step planning can still work for streaming-style sources without materializing them.

#### File: `easydel/data/sources/hf_wrapper.py`

Extend `HFDatasetShardedSource` similarly.

Required behavior:

- for iterable datasets, attempt to derive `_length` from attached split metadata
- expose that value through `estimated_length`
- keep `__len__` raising when true length is unavailable, but allow `estimated_length` to exist independently

## 2. Loss masking must survive packing

### Motivation

In local supervised fine-tuning, masking the prompt or non-assistant tokens is not optional. The branch depends on "assistant-only" or "completion-only" loss continuing to mean the same thing after tokenization and after packing.

Upstream behavior is closer to "mask information exists during preprocessing". The local branch requires "explicit labels survive the whole preprocessing pipeline".

### Behavioral target

After reproducing this divergence:

- the preprocessing stage should emit `labels` with `-100` on ignored positions
- packing should preserve those labels and keep them aligned with `input_ids`
- assistant-only loss and completion-only loss should still behave correctly after packed training examples are formed

### Reconstruction steps

#### File: `easydel/trainers/prompt_transforms.py`

Add label construction at preprocessing time.

Required behavior:

- add a helper like `_apply_completion_labels(result: dict) -> None`
- it should:
  - return early unless prompt masking is active
  - look for `completion_mask`
  - if absent, fall back to `assistant_masks`
  - combine that mask with `attention_mask` when present
  - create `labels` such that:
    - kept positions get the token id
    - ignored or padded positions get `-100`

Call that helper:

- after tokenizing conversational or prompt/completion examples
- after any branch where completion masking is produced

The local branch specifically converts masking into explicit labels before the packer sees the sample.

#### File: `easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py`

Change the meaning of prompt masking for SFT.

Required behavior:

- when building `SFTPreprocessTransform`, set `mask_prompt` true if either:
  - `assistant_only_loss` is true
  - `completion_only_loss` is true

This differs from upstream because the local branch treats assistant-only loss as another path to the same masking semantics.

There is also a packing-related preprocessing difference:

- `SFTPreprocessTransform` should accept `pad_to_max_length: bool = True`
- when packing is enabled, the SFT trainer should pass `pad_to_max_length=False`
- tokenization should then use a helper like `_padding_mode()` so examples are not pre-padded before packing

This is part of the local divergence because the branch expects masked-label preprocessing and packing to work together on unpadded token sequences.

#### File: `easydel/data/transforms/pack.py`

Teach all packers to preserve `labels` and emit packed-loss metadata.

Required behavior:

- `PackedSequence` must gain:
  - `labels: np.ndarray | None = None`
- `PackedSequence` must also gain:
  - `decoder_positions: np.ndarray | None = None`
- `PackedSequence.to_dict()` must include `labels` when present
- `PackedSequence.to_dict()` must also expose:
  - `decoder_segment_ids`
  - `decoder_positions`

Update `GreedyPacker`:

- maintain an internal `_labels` buffer alongside `_buffer`
- change `add(...)` signature to accept `labels: list[int] | None`
- require labels to match token length when provided
- normalize each sample before packing:
  - clip to `seq_length`
  - append EOS only if the sample is non-empty, does not already end with EOS, and still has room
- if labels exist for the packer state, append `-100` for any EOS inserted during normalization
- `_flush()` must return aligned `labels`
- `flush_final()` must:
  - truncate to `seq_length`
  - pad labels with `-100`
  - pad `segment_ids` with `-1`
  - generate per-segment-reset `decoder_positions`

Update `PoolPacker`:

- forward optional labels into the chosen inner packer
- estimate fit using normalized sample length rather than assuming a fresh EOS is always appended

Update `FirstFitPacker`:

- store pending items as `(tokens, labels, source_id)`
- normalize pending samples before sorting and bin-packing
- when building bins, merge labels alongside tokens
- preserve labels exactly as produced by normalization
- pad labels with `-100`
- pad `segment_ids` with `-1`
- generate `decoder_positions`

Update `PackedShardedSource`:

- read `labels = example.get("labels")`
- pass labels into whichever packer is active
- keep iterating the underlying source's real shard names; do not forward the synthetic `"packed_shard_0"` name into the wrapped source

The local branch depends on this exact propagation path: preprocess produces labels, pack preserves labels, trainer consumes labels.

## 3. The local workflow needs a direct finetune entrypoint

### Motivation

Upstream provides reusable trainer infrastructure. The local branch also needs a single operational script that directly expresses the most common training workflow used in this repository.

This is not a framework-level abstraction change. It is an explicit workflow entrypoint.

### Behavioral target

After reproducing this divergence, the repository should contain one script that:

- initializes JAX distributed runtime
- parses an SFT config plus runtime config
- loads tokenizer and dataset
- chooses model loader based on architecture
- supports `full`, `lora`, and `lora_embed_head` training modes
- runs training
- merges LoRA if needed
- exports an HF-compatible model in a multi-host-safe way

### Reconstruction steps

#### File: `easydel/scripts/finetune/train.py`

Create a new script with the following structure:

- call `jax.distributed.initialize()`
- define a runtime dataclass with fields for:
  - `repo_id`
  - `training_mode`
  - `lora_rank`
  - `lora_pattern`
  - dataset name / split / subset / streaming / cache dir
  - processor repo id
  - sharding axis and optional DCN sharding axis
  - attention mechanism
  - gradient checkpointing
  - dtype / param_dtype / attn_dtype / attn_softmax_dtype
- parse `(ed.SFTConfig, RunTimeConfig)` using `DataClassArgumentParser`
- load tokenizer and patch `pad_token_id` from `eos_token_id` when missing
- build dataset with `HuggingFaceShardedSource`
- inspect HF config and choose:
  - `AutoEasyDeLModelForImageTextToText`
  - or `AutoEasyDeLModelForCausalLM`
- construct the model with local sharding/runtime settings

Training mode behavior:

- `full`: train normally
- `lora`: call `apply_lora_to_layers(...)`
- `lora_embed_head`:
  - apply LoRA
  - additionally convert embedding weights and lm_head weights into trainable `nn.LoRAParam`

Formatting behavior:

- inspect a dataset sample
- if the target text field contains message-style data, leave formatting as `None`
- let `SFTPreprocessTransform` apply the tokenizer chat template directly
- this preserves conversational structure so `assistant_only_loss` can request assistant masks from the tokenizer
- if the target field is plain text, leave formatting function as `None`

Training behavior:

- instantiate `ed.SFTTrainer`
- run `trainer.train()`
- recover final state from output or trainer state

Export behavior:

- unwrap LoRA if used
- restore embedding / lm_head params from `LoRAParam` back to plain `Param` when needed
- set:
  - `EASY_SAFE_TRANSFER=1`
  - `EASYDEL_CHUNK_BYTES=64 * 1024 * 1024`
- synchronize devices before export
- call `StateDictConverter.easydel_to_torch(...)` on all hosts
- only on process 0:
  - instantiate the target HF model on `torch.device("meta")`
  - load the converted state dict with `assign=True`
  - save with `safe_serialization=True`
- synchronize again after save

This script is part of the local divergence because it codifies the preferred local workflow instead of leaving the workflow to external glue code.

## 4. Weight export must be more robust on multi-host TPU systems

### Motivation

The local environment cares about successful export in distributed settings more than about keeping the conversion code minimal.

The upstream conversion path is not sufficient for the local branch's export expectations on large sharded models.

### Behavioral target

After reproducing this divergence:

- JAX arrays should be exportable even when only locally addressable shards are visible
- multi-host TPU export should gather full arrays onto process 0
- large arrays should be movable in chunks to reduce peak memory
- state-dict conversion should attempt per-parameter gather logic before converting tensors

### Reconstruction steps

#### File: `easydel/utils/parameters_transformation.py`

Import:

- `from jax.experimental import multihost_utils as mhutils`

Inside `TensorConverter.jax_to_pytorch(...)`, add:

- `_get_local_array(arr)`
  - prefer fully addressable arrays
  - otherwise prefer first addressable shard's data
- platform detection based on:
  - `jax.devices()`
  - `jax.default_backend()`
- `_cpu_chunked_transfer(arr)`
  - flatten to 1D
  - move chunks sized by `EASYDEL_CHUNK_BYTES`
  - reassemble on CPU
  - convert bfloat16 safely for torch

Add a TPU-specialized path:

- if backend is TPU:
  - call `global_array_to_host_numpy(...)`
  - on process 0, return a real torch tensor
  - on non-main processes, return a placeholder tensor

Add a "safe transfer" path:

- if `EASY_SAFE_TRANSFER` is true:
  - use chunked CPU transfer instead of naive `device_get(...).tolist()`

Keep a DLPack-based path for the non-safe case, using a helper like `_to_dlpack_capsule(arr)`.

Also add:

- `global_array_to_host_numpy(x, target_dtype)`
  - fast path for fully addressable arrays
  - otherwise collect `addressable_shards`
  - store shard index ranges plus host arrays
  - `mhutils.process_allgather(...)`
  - on process 0, reconstruct the full numpy array from gathered shard payloads
  - on other processes, return `None`

Update `StateDictConverter.easydel_to_torch(...)`:

- attempt to read `module._gather_fns`
- flatten that tree
- if a gather function exists for a parameter key, apply it before conversion

This is meant to avoid exporting only a local shard of a parameter when the local branch expects a full global parameter.

Update `ModelConverter.to_torch(...)`:

- if a key expected by the target HF model is missing from the converted state dict, warn instead of indexing blindly
- keep the local branch's debug printing behavior:
  - print `"state_dict:"`
  - then print every key before loading

That debug output is part of the current divergence and should be reproduced if exact parity is required.

## 5. LoRA-wrapped layers must preserve sharding semantics

### Motivation

In the local branch, LoRA is not just an adapter mechanism. It must remain compatible with distributed partitioning decisions already encoded in `ParallelLinear`.

### Behavioral target

After reproducing this divergence:

- applying LoRA to a `ParallelLinear` should still expose usable sharding metadata
- LoRA A/B matrices should be assigned sharding specs based on whether the wrapped linear layer is row-parallel or column-parallel
- the wrapped base module's own sharding information should still be available

### Reconstruction steps

#### File: `easydel/infra/utils.py`

Inside `apply_lora_to_layers(...)`:

- instead of directly constructing and inserting `nn.LoRA(...)` inline, first build it into a variable, for example `lora_module`
- dynamically attach a `craft_sharding(...)` method to that LoRA module

That `craft_sharding(...)` must:

- inspect `self.base_module`
- read the wrapped module's `_direction`
- choose LoRA sharding as:
  - row-parallel base:
    - `lora_a -> RowWise`
    - `lora_b -> Replicated`
  - column-parallel base:
    - `lora_a -> Replicated`
    - `lora_b -> ColumnWise`
  - otherwise:
    - both replicated
- call `resolve_safe_sharding(...)` for `lora_a` and `lora_b`
- if the base module also has `craft_sharding(...)`, include its entries in the returned spec map under a `base_module/...` prefix

Then insert that `lora_module` into the model with `set_module_from_path(...)`.

This is the local branch's way of ensuring that adapter insertion does not erase the distributed layout model.

## 6. Trainer step functions must merge auxiliary graph state explicitly

### Motivation

The local branch needs train and eval steps to reconstruct the module with more than just the trainable graph state. Some auxiliary graph state must survive merge boundaries but should not receive gradients.

### Behavioral target

After reproducing this divergence:

- train and eval step functions should merge:
  - `graphdef`
  - the current trainable tree
  - the auxiliary `graphother` tree
- `graphother` must be wrapped in `stop_gradient`

### Reconstruction steps

#### File: `easydel/trainers/trainer/_fn.py`

In both `training_step(...)` and `evaluation_step(...)`:

- replace the local merge pattern `module = state.merge(tree)`
- construct:
  - `tree_other = tree_map(lambda x: stop_gradient(asarray(x)) if hasattr(x, "shape") else x, state.graphother)`
- then call:
  - `nn.merge(state.graphdef, tree, tree_other)`

This change should be applied symmetrically in train and eval code paths.

## 7. The local GLM implementation must match local expectations better than upstream does

### Motivation

The local branch carries a GLM-specific compatibility patch. The goal is not to create a more generic GLM implementation. The goal is to make the GLM module structure align with local training and weight expectations.

### Behavioral target

After reproducing this divergence:

- GLM MLP structure should use separate gate and up projections rather than a fused gate-up projection
- GLM attention projections should use local bias choices
- GLM rotary embedding creation should follow the local configuration path

### Reconstruction steps

#### File: `easydel/modules/glm/modeling_glm.py`

Change the MLP structure:

- replace a fused `gate_up_proj` projection with:
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- in the forward pass:
  - compute `gate = act_fn(gate_proj(hidden_states))`
  - compute `up = up_proj(hidden_states)`
  - multiply `gate * up`
  - pass through `down_proj`
- keep checkpoint labels around these operations

Change attention projection creation:

- override `_create_q_proj(...)` to use `use_bias=True`
- override `_create_k_proj(...)` to use `use_bias=True`
- override `_create_v_proj(...)` to use `use_bias=True`
- override `_create_o_proj(...)` to use `use_bias=False`

Change rotary construction:

- override `_create_rotary(...)`
- call `config.get_basic_rope(...)`
- use:
  - `head_size=self.head_dim`
  - `rotary_dim=self.head_dim`
  - `base=getattr(config, "rope_theta", 100000000.0)`
  - `is_neox_style=False`

This file represents a local model-compatibility patch, not a general upstream design preference.

## 8. Local attention kernels should not force `logits_dtype`

### Motivation

The local branch carries a narrow kernel-level compatibility tweak: do not force `logits_dtype` at the flash-attention call site.

### Behavioral target

After reproducing this divergence:

- the flash-attention call path should no longer explicitly pass `logits_dtype=jnp.bfloat16`

### Reconstruction steps

#### File: `easydel/operations/kernels/flash_attention.py`

At the flash attention invocation site:

- remove or comment out the explicit `logits_dtype=jnp.bfloat16` argument

#### File: `easydel/operations/kernels/paged_flash_attention.py`

Do the same for paged flash attention:

- remove or comment out the explicit `logits_dtype=jnp.bfloat16` argument

The local branch leaves the rest of the attention call structure unchanged.

## 9. Configuration typing is locally tightened

### Motivation

The local branch slightly narrows configuration shapes to match the local calling convention more closely.

### Behavioral target

After reproducing this divergence:

- the typed trainer config should treat `track_memory` as `bool`, not `bool | float`

### Reconstruction steps

#### File: `easydel/infra/elarge_model/trainer_types.py`

Change:

- `track_memory: NotRequired[bool | float]`

to:

- `track_memory: NotRequired[bool]`

This is a local typing-level divergence, separate from the training logic changes described earlier.

## 10. A low-impact repository-level script mode difference remains

### Motivation

One tracked difference is not semantic. It is a file mode difference on a helper shell script.

### Behavioral target

After reproducing this divergence:

- the file mode of the script should match the local branch rather than upstream

### Reconstruction steps

#### File: `scripts/mount_gcsfuse.sh`

Reproduce the mode change:

- upstream mode: executable
- local mode: non-executable

No content change is required for this file.

## 11. Packed segment metadata must also drive model-side mask preparation

### Motivation

The local branch expects packed examples to remain usable after they leave the data pipeline. Carrying `segment_ids` alone is not sufficient if model input preparation does not convert them into the attention-mask metadata expected by downstream kernels.

This divergence is therefore not just about data packing. It also ensures packed segment boundaries are converted into model-consumable mask structure automatically.

### Behavioral target

After reproducing this divergence:

- model input preparation should notice `segment_ids`
- if `mask_info` is missing, it should be derived from `segment_ids`
- callers that pass packed samples with segment ids should not need to build `mask_info` manually

### Reconstruction steps

#### File: `easydel/infra/base_module.py`

Update `EasyDeLBaseModule.prepare_inputs_for_call(...)`.

Required behavior:

- import `MaskInfo` from `ejkernel.types`
- read `segment_ids` from `kwargs`
- if `segment_ids` is present and `mask_info` is not already provided:
  - build `MaskInfo.from_segments(jnp.asarray(segment_ids, dtype=jnp.int32))`
  - assign it to `kwargs["mask_info"]`

This is a small patch, but it is part of the actual upstream divergence because it connects packed sample metadata to the model-side attention path.

## 12. Summary of what must be true if the divergence is reproduced correctly

A correct reconstruction of the current local branch should produce all of the following outcomes:

- training can be limited by raw consumed examples
- raw-item progress survives checkpoint save/load
- progress bars and metrics can reflect raw-item progress
- iterable Hugging Face datasets can contribute metadata-derived length information
- preprocessing emits explicit `labels` for masked SFT loss
- packing preserves those labels and keeps them aligned
- packed padding uses `segment_ids=-1`
- packed sequences also expose `decoder_segment_ids` and `decoder_positions` for packed-aware loss handling
- packed `segment_ids` can automatically become model-side `mask_info`
- a local end-to-end finetune script exists and supports LoRA-oriented workflows
- JAX-to-PyTorch export is adapted for multi-host TPU usage
- LoRA insertion preserves usable sharding semantics
- trainer step functions explicitly merge auxiliary graph state
- GLM modeling differs from upstream in a local compatibility-oriented way
- flash attention call sites no longer force `logits_dtype`
- local typing narrows `track_memory`
- the repository preserves the local shell-script mode difference

If any of those outcomes are missing, the local divergence has only been partially reproduced.
