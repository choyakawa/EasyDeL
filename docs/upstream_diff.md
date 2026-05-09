# Local Divergence From `upstream/main`

This document describes the branch's real code-level divergence from `upstream/main` as it exists now.

It is meant for a maintainer who wants to reconstruct the local behavior starting from upstream without replaying local history.

This guide intentionally focuses on behavioral and structural differences. It does not try to document merge commits, and it ignores low-signal metadata-only differences such as the executable-bit change on `scripts/mount_gcsfuse.sh`.

## 1. Training config memory tracking is typed more narrowly

### Motivation

The local branch narrows the trainer memory-tracking configuration surface.

### Behavioral target

Starting from upstream, reproduce this property:

- `track_memory` accepts a boolean value rather than `bool | float`

### Reconstruction steps

#### File: `easydel/trainers/training_configurations.py`

- `track_memory` is narrowed from `bool | float` to `bool`

#### File: `easydel/infra/elarge/types/training.py`

Mirror the same typing change in the typed config surface:

- `track_memory: NotRequired[bool]`

## 2. The repository includes a local finetune entrypoint

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

## 3. Export and LoRA wrapping are customized for multi-host and sharded local workflows

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

## 4. Trainer step functions explicitly merge auxiliary graph state

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

## 5. The local GLM implementation intentionally diverges from upstream

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

## 6. Local attention kernels do not force `logits_dtype`

### Motivation

The local branch avoids hard-coding `logits_dtype=jnp.bfloat16` in these flash-attention call sites.

### Behavioral target

The kernels should inherit the upstream/default dtype behavior instead of forcing bf16 logits.

### Reconstruction steps

#### File: `easydel/operations/kernels/flash_attention.py`

- remove the forced `logits_dtype=jnp.bfloat16` argument from the flash-attention invocation

#### File: `easydel/operations/kernels/paged_flash_attention.py`

- remove the forced `logits_dtype=jnp.bfloat16` argument from the paged flash-attention invocation

## 7. Summary

If the local divergence has been reproduced correctly, these statements should all be true:

- `track_memory` is narrowed to boolean-only typing
- the repository contains a direct SFT entrypoint script for the local workflow
- JAX-to-torch export is multi-host-safe and chunked for large arrays
- LoRA wrappers preserve EasyDeL calling conventions and sharding metadata
- trainer step functions merge `graphother` explicitly
- the local GLM implementation and flash-attention call sites intentionally differ from upstream
