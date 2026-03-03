from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from eformer.aparser import DataClassArgumentParser
from transformers import AutoTokenizer

from easydel.data import HuggingFaceShardedSource
from easydel.data.transforms.pack import PackedShardedSource
from easydel.infra.loss_utils import _shifted_segment_continuation_mask
from easydel.trainers.prompt_transforms import SFTPreprocessTransform

try:
    from ejkernel.types import MaskInfo
except ImportError:
    from ejkernel.types.mask import MaskInfo  # pyright: ignore[reportMissingImports]


@dataclass
class InspectPackingConfig:
    repo_id: str | None = field(
        default=None,
        metadata={"help": "Optional model repo ID. Only used to default processor_repo_id."},
    )
    processor_repo_id: str | None = field(
        default=None,
        metadata={"help": "Tokenizer/processor repo ID. Defaults to repo_id."},
    )
    dataset_name: str = field(
        default="trl-lib/Capybara",
        metadata={"help": "Dataset name or local path."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to inspect."},
    )
    dataset_subset: str | None = field(
        default=None,
        metadata={"help": "Dataset subset/config."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset."},
    )
    dataset_cache_dir: str | None = field(
        default=None,
        metadata={"help": "Optional dataset cache directory."},
    )
    dataset_text_field: str | None = field(
        default="text",
        metadata={"help": "Dataset text field. For chat data, pass messages/conversations field if needed."},
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Target sequence length used by SFT preprocessing/packing."},
    )
    packing: bool = field(
        default=True,
        metadata={"help": "Whether to inspect packed data path."},
    )
    packing_strategy: str = field(
        default="bfd",
        metadata={"help": "Packing strategy from SFTConfig: bfd or wrapped."},
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={"help": "Mirror SFT assistant_only_loss."},
    )
    completion_only_loss: bool | None = field(
        default=None,
        metadata={"help": "Deprecated alias. If set, ORed with assistant_only_loss."},
    )
    add_eos: bool = field(
        default=True,
        metadata={"help": "Whether SFT preprocessing appends EOS to plain text rows."},
    )
    raw_examples: int = field(
        default=3,
        metadata={"help": "How many tokenized-but-unpacked examples to print."},
    )
    packed_examples: int = field(
        default=3,
        metadata={"help": "How many packed examples to print."},
    )
    boundary_window: int = field(
        default=8,
        metadata={"help": "How many tokens to show around each packed boundary."},
    )
    max_boundaries_per_example: int = field(
        default=4,
        metadata={"help": "Max boundary snippets printed per packed example."},
    )
    decode_tokens: bool = field(
        default=False,
        metadata={"help": "Decode token windows with the tokenizer for readability."},
    )
    dump_json_path: str | None = field(
        default=None,
        metadata={"help": "Optional path to write a JSON summary."},
    )

    def __post_init__(self):
        if self.processor_repo_id is None:
            self.processor_repo_id = self.repo_id
        if self.processor_repo_id is None:
            raise ValueError("processor_repo_id or repo_id must be provided.")


def _take_examples(source, limit: int) -> list[dict]:
    examples: list[dict] = []
    for shard_name in source.shard_names:
        for example in source.open_shard(shard_name):
            examples.append(example)
            if len(examples) >= limit:
                return examples
    return examples


def _safe_decode(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return "<decode failed>"


def _array_to_list(value) -> list[int] | None:
    if value is None:
        return None
    return np.asarray(value).astype(np.int32, copy=False).tolist()


def _format_snippet(
    tokenizer,
    input_ids: np.ndarray,
    labels: np.ndarray,
    segment_ids: np.ndarray,
    position_ids: np.ndarray,
    effective_loss_mask: np.ndarray,
    center: int,
    window: int,
    decode_tokens: bool,
) -> dict[str, object]:
    start = max(0, center - window)
    end = min(input_ids.shape[0], center + window + 1)
    snippet = {
        "span": [int(start), int(end)],
        "input_ids": input_ids[start:end].tolist(),
        "labels": labels[start:end].tolist(),
        "segment_ids": segment_ids[start:end].tolist(),
        "position_ids": position_ids[start:end].tolist(),
        "effective_loss_mask": effective_loss_mask[start:end].astype(np.int32).tolist(),
    }
    if decode_tokens:
        snippet["decoded_text"] = _safe_decode(tokenizer, input_ids[start:end].tolist())
    return snippet


def _inspect_tokenized_examples(tokenizer, examples: list[dict]) -> list[dict[str, object]]:
    inspected: list[dict[str, object]] = []
    for idx, example in enumerate(examples):
        input_ids = np.asarray(example["input_ids"], dtype=np.int32)
        attention_mask = np.asarray(example.get("attention_mask", np.ones_like(input_ids)), dtype=np.int32)
        labels = example.get("labels")
        label_array = np.asarray(labels, dtype=np.int32) if labels is not None else None

        item: dict[str, object] = {
            "example_index": idx,
            "input_length": int(input_ids.shape[0]),
            "valid_tokens": int(attention_mask.sum()),
            "has_labels": labels is not None,
            "first_tokens": input_ids[: min(24, input_ids.shape[0])].tolist(),
        }
        if label_array is not None:
            item["supervised_tokens"] = int(np.count_nonzero(label_array != -100))
            item["first_labels"] = label_array[: min(24, label_array.shape[0])].tolist()
        inspected.append(item)
    return inspected


def _inspect_packed_example(
    tokenizer,
    example: dict,
    example_index: int,
    boundary_window: int,
    max_boundaries_per_example: int,
    decode_tokens: bool,
) -> dict[str, object]:
    input_ids = np.asarray(example["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(example.get("attention_mask", np.ones_like(input_ids)), dtype=np.int32)
    labels = example.get("labels")
    label_array = np.asarray(labels, dtype=np.int32) if labels is not None else input_ids.copy()
    segment_ids = np.asarray(example["segment_ids"], dtype=np.int32)

    mask_info = MaskInfo.from_segments(segment_ids[None, :])
    mask_position_ids = np.asarray(mask_info.q_position_ids)[0].astype(np.int32, copy=False)

    decoder_positions = example.get("decoder_positions")
    decoder_positions_array = (
        np.asarray(decoder_positions, dtype=np.int32) if decoder_positions is not None else None
    )

    shift_labels = label_array[1:]
    shift_attention_mask = attention_mask[1:].astype(bool)
    continuation_mask = np.asarray(_shifted_segment_continuation_mask(segment_ids), dtype=bool)[0]
    effective_shift_loss_mask = shift_attention_mask & continuation_mask & (shift_labels != -100)

    effective_loss_mask = np.zeros_like(input_ids, dtype=bool)
    effective_loss_mask[1:] = effective_shift_loss_mask

    boundary_positions = np.flatnonzero(
        (segment_ids[:-1] >= 0) & (segment_ids[1:] >= 0) & (segment_ids[:-1] != segment_ids[1:])
    )
    boundary_starts = boundary_positions + 1
    boundary_loss_mask_nonzero = int(np.count_nonzero(effective_shift_loss_mask[boundary_positions]))
    boundary_target_unmasked = int(np.count_nonzero(shift_labels[boundary_positions] != -100))

    position_reset_failures = [
        int(pos)
        for pos in boundary_starts
        if pos < mask_position_ids.shape[0] and int(mask_position_ids[pos]) != 0
    ]

    valid_position_match = None
    if decoder_positions_array is not None:
        valid_mask = segment_ids >= 0
        valid_position_match = bool(np.array_equal(decoder_positions_array[valid_mask], mask_position_ids[valid_mask]))

    boundary_snippets = []
    for pos in boundary_positions[:max_boundaries_per_example]:
        boundary_snippets.append(
            {
                "transition_at": int(pos),
                "from_segment": int(segment_ids[pos]),
                "to_segment": int(segment_ids[pos + 1]),
                "target_token_after_boundary": int(shift_labels[pos]),
                "continuation_mask": int(continuation_mask[pos]),
                "effective_shift_loss_mask": int(effective_shift_loss_mask[pos]),
                "window": _format_snippet(
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    labels=label_array,
                    segment_ids=segment_ids,
                    position_ids=mask_position_ids,
                    effective_loss_mask=effective_loss_mask,
                    center=int(pos),
                    window=boundary_window,
                    decode_tokens=decode_tokens,
                ),
            }
        )

    return {
        "example_index": example_index,
        "packed_length": int(input_ids.shape[0]),
        "valid_tokens": int(attention_mask.sum()),
        "segment_count": int(np.unique(segment_ids[segment_ids >= 0]).size),
        "boundary_count": int(boundary_positions.size),
        "boundary_target_unmasked_count": boundary_target_unmasked,
        "boundary_effective_loss_nonzero_count": boundary_loss_mask_nonzero,
        "supervised_token_count": int(np.count_nonzero(label_array != -100)),
        "position_reset_failure_positions": position_reset_failures,
        "decoder_positions_match_maskinfo": valid_position_match,
        "first_segment_ids": segment_ids[: min(32, segment_ids.shape[0])].tolist(),
        "first_position_ids": mask_position_ids[: min(32, mask_position_ids.shape[0])].tolist(),
        "first_labels": label_array[: min(32, label_array.shape[0])].tolist(),
        "boundary_snippets": boundary_snippets,
    }


def main():
    parser = DataClassArgumentParser((InspectPackingConfig,))
    (args,) = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(args.processor_repo_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = HuggingFaceShardedSource(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        subset=args.dataset_subset,
        streaming=args.streaming,
        cache_dir=args.dataset_cache_dir,
    )

    try:
        sample = next(dataset.open_shard(dataset.shard_names[0]))
    except StopIteration as exc:
        raise ValueError("Dataset is empty; cannot inspect packing.") from exc

    pretokenized = "input_ids" in sample
    mask_prompt = bool(args.assistant_only_loss)
    if args.completion_only_loss is not None:
        mask_prompt = mask_prompt or bool(args.completion_only_loss)

    tokenized_source = dataset
    if not pretokenized:
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=args.max_length,
            text_field=args.dataset_text_field or "text",
            mask_prompt=mask_prompt,
            add_eos=args.add_eos,
            pad_to_max_length=not args.packing,
            formatting_func=None,
        )
        tokenized_source = dataset.transform(transform)

    tokenized_examples = _take_examples(tokenized_source, args.raw_examples)

    summary: dict[str, object] = {
        "config": {
            "repo_id": args.repo_id,
            "processor_repo_id": args.processor_repo_id,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "dataset_subset": args.dataset_subset,
            "streaming": args.streaming,
            "dataset_text_field": args.dataset_text_field,
            "max_length": args.max_length,
            "packing": args.packing,
            "packing_strategy": args.packing_strategy,
            "assistant_only_loss": args.assistant_only_loss,
            "completion_only_loss": args.completion_only_loss,
        },
        "source_sample_keys": sorted(sample.keys()),
        "pretokenized": pretokenized,
        "tokenized_examples": _inspect_tokenized_examples(tokenizer, tokenized_examples),
    }

    if args.packing:
        strategy_map = {"bfd": "first_fit", "wrapped": "greedy"}
        packed_source = PackedShardedSource(
            source=tokenized_source,
            seq_length=args.max_length,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id or 0,
            strategy=strategy_map.get(args.packing_strategy, "greedy"),
            include_segment_ids=True,
            shuffle=False,
        )
        packed_examples = _take_examples(packed_source, args.packed_examples)
        packed_inspection = [
            _inspect_packed_example(
                tokenizer=tokenizer,
                example=example,
                example_index=idx,
                boundary_window=args.boundary_window,
                max_boundaries_per_example=args.max_boundaries_per_example,
                decode_tokens=args.decode_tokens,
            )
            for idx, example in enumerate(packed_examples)
        ]
        summary["packed_examples"] = packed_inspection
        summary["packed_rollup"] = {
            "packed_examples_inspected": len(packed_inspection),
            "total_boundaries": int(sum(item["boundary_count"] for item in packed_inspection)),
            "total_boundary_target_unmasked": int(
                sum(item["boundary_target_unmasked_count"] for item in packed_inspection)
            ),
            "total_boundary_effective_loss_nonzero": int(
                sum(item["boundary_effective_loss_nonzero_count"] for item in packed_inspection)
            ),
            "total_position_reset_failures": int(
                sum(len(item["position_reset_failure_positions"]) for item in packed_inspection)
            ),
        }

    if args.dump_json_path is not None:
        with open(args.dump_json_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=True, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
