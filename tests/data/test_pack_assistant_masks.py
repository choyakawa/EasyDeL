from __future__ import annotations

from collections.abc import Iterator, Sequence

import jax.numpy as jnp
import numpy as np
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]

from easydel.data.core.protocols import ShardedDataSource
from easydel.data.transforms.pack import PackedShardedSource


class _ListSource(ShardedDataSource[dict]):
    def __init__(self, rows: list[dict]):
        self._rows = rows

    @property
    def shard_names(self) -> Sequence[str]:
        return ["shard"]

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        del shard_name
        yield from self._rows

    def __len__(self) -> int:
        return len(self._rows)


class _StreamingListSource(ShardedDataSource[dict]):
    def __init__(self, rows: list[dict]):
        self._rows = rows

    @property
    def shard_names(self) -> Sequence[str]:
        return ["streaming_shard"]

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        del shard_name
        yield from self._rows


def test_packed_source_preserves_assistant_masks_across_segments():
    source = _ListSource(
        [
            {
                "input_ids": [10, 11, 12],
                "attention_mask": [1, 1, 1],
                "assistant_masks": [0, 1, 1],
                "completion_mask": [0, 1, 1],
            },
            {
                "input_ids": [20, 21],
                "attention_mask": [1, 1],
                "assistant_masks": [0, 1],
                "completion_mask": [0, 1],
            },
        ]
    )

    packed = PackedShardedSource(
        source,
        seq_length=8,
        eos_token_id=99,
        pad_token_id=0,
        strategy="first_fit",
        shuffle=False,
        aligned_fields=("attention_mask", "assistant_masks", "completion_mask"),
    )

    row = next(packed.open_shard("packed_shard_0"))

    np.testing.assert_array_equal(row["input_ids"], np.array([10, 11, 12, 99, 20, 21, 99, 0]))
    np.testing.assert_array_equal(row["attention_mask"], np.array([1, 1, 1, 1, 1, 1, 1, 0]))
    np.testing.assert_array_equal(row["segment_ids"], np.array([1, 1, 1, 1, 2, 2, 2, 0]))
    np.testing.assert_array_equal(row["assistant_masks"], np.array([0, 1, 1, 0, 0, 1, 0, 0]))
    np.testing.assert_array_equal(row["completion_mask"], np.array([0, 1, 1, 0, 0, 1, 0, 0]))
    np.testing.assert_array_equal(row["position_ids"], np.array([0, 1, 2, 3, 0, 1, 2, 0]))
    assert int(row["__num_source_examples"]) == 2


def test_packed_source_strips_existing_padding_before_packing_masks():
    source = _ListSource(
        [
            {
                "input_ids": [0, 0, 10, 11],
                "attention_mask": [0, 0, 1, 1],
                "assistant_masks": [0, 0, 0, 1],
            },
        ]
    )

    packed = PackedShardedSource(
        source,
        seq_length=5,
        eos_token_id=99,
        pad_token_id=0,
        strategy="first_fit",
        shuffle=False,
        aligned_fields=("attention_mask", "assistant_masks"),
    )

    row = next(packed.open_shard("packed_shard_0"))

    np.testing.assert_array_equal(row["input_ids"], np.array([10, 11, 99, 0, 0]))
    np.testing.assert_array_equal(row["attention_mask"], np.array([1, 1, 1, 0, 0]))
    np.testing.assert_array_equal(row["assistant_masks"], np.array([0, 1, 0, 0, 0]))


def test_packed_source_iterates_streaming_source_without_length():
    source = _StreamingListSource(
        [
            {
                "input_ids": [10, 11],
                "attention_mask": [1, 1],
                "assistant_masks": [0, 1],
            },
            {
                "input_ids": [20],
                "attention_mask": [1],
                "assistant_masks": [1],
            },
        ]
    )

    packed = PackedShardedSource(
        source,
        seq_length=6,
        eos_token_id=99,
        pad_token_id=0,
        strategy="first_fit",
        shuffle=False,
        aligned_fields=("attention_mask", "assistant_masks"),
    )

    row = next(packed.open_shard("packed_shard_0"))

    np.testing.assert_array_equal(row["input_ids"], np.array([10, 11, 99, 20, 99, 0]))
    np.testing.assert_array_equal(row["attention_mask"], np.array([1, 1, 1, 1, 1, 0]))
    np.testing.assert_array_equal(row["assistant_masks"], np.array([0, 1, 0, 1, 0, 0]))
    assert int(row["__num_source_examples"]) == 2


def test_packed_segment_ids_create_block_diagonal_attention_mask():
    segment_ids = jnp.array([[1, 1, 1, 1, 2, 2, 2, 0]], dtype=jnp.int32)
    position_ids = jnp.array([[0, 1, 2, 3, 0, 1, 2, 0]], dtype=jnp.int32)
    mask_info = MaskInfo.from_segments(
        q_segment_ids=segment_ids,
        kv_segment_ids=segment_ids,
        q_positions=position_ids,
        kv_positions=position_ids,
    )

    attention = np.asarray(mask_info.get_or_compute_attention_mask(dtype=jnp.bool_))
    attention = np.squeeze(attention, axis=tuple(range(attention.ndim - 2)))

    assert not attention[:4, 4:7].any()
    assert not attention[4:7, :4].any()
    assert not attention[7, :].any()
    assert not attention[:, 7].any()
    assert attention[:4, :4].any()
    assert attention[4:7, 4:7].any()
