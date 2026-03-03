# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Token packing utilities for efficient training.

This module provides:
- Greedy packing: Simple concatenation-based packing
- Pool-based packing: Multiple packers for efficient bin-packing
- First-fit packing: Bin-packing with first-fit decreasing
- Segment IDs for attention masking
"""

from __future__ import annotations

import logging
import random
import typing as tp
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..core.config import PackStageConfig
from ..core.protocols import BaseStage, PipelineContext, ShardedDataSource

if tp.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)


_PADDING_SEGMENT_ID = -1


def _build_decoder_positions(segment_ids: np.ndarray | None) -> np.ndarray | None:
    """Build per-segment reset positions for packed loss normalization."""
    if segment_ids is None:
        return None

    positions = np.zeros_like(segment_ids, dtype=np.int32)
    current_segment: int | None = None
    current_position = 0

    for idx, segment_id in enumerate(segment_ids.tolist()):
        if segment_id == _PADDING_SEGMENT_ID:
            positions[idx] = 0
            current_segment = None
            current_position = 0
            continue

        if segment_id != current_segment:
            current_segment = segment_id
            current_position = 0

        positions[idx] = current_position
        current_position += 1

    return positions


def _normalize_sample(
    tokens: list[int],
    labels: list[int] | None,
    eos_token_id: int,
    seq_length: int,
) -> tuple[list[int], list[int] | None]:
    """Normalize a pre-tokenized sample before packing.

    Ensures labels stay aligned and avoids introducing a second EOS when the
    sample already ends with one. EOS is appended only when there is space.
    """
    if labels is not None and len(labels) != len(tokens):
        raise ValueError("labels must be aligned with tokens when packing")

    normalized_tokens = list(tokens[:seq_length])
    normalized_labels = list(labels[:seq_length]) if labels is not None else None

    should_append_eos = (
        bool(normalized_tokens)
        and normalized_tokens[-1] != eos_token_id
        and len(normalized_tokens) < seq_length
    )
    if should_append_eos:
        normalized_tokens.append(eos_token_id)
        if normalized_labels is not None:
            normalized_labels.append(-100)

    return normalized_tokens, normalized_labels


@dataclass
class PackedSequence:
    """A packed sequence with metadata."""

    input_ids: np.ndarray
    labels: np.ndarray | None = None
    attention_mask: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    decoder_positions: np.ndarray | None = None
    source_ids: list[str] | None = None
    num_segments: int = 0

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary for training."""
        result = {"input_ids": self.input_ids}
        if self.labels is not None:
            result["labels"] = self.labels
        if self.attention_mask is not None:
            result["attention_mask"] = self.attention_mask
        if self.segment_ids is not None:
            result["segment_ids"] = self.segment_ids
            result["decoder_segment_ids"] = self.segment_ids
        if self.decoder_positions is not None:
            result["decoder_positions"] = self.decoder_positions
        return result


class GreedyPacker:
    """Simple greedy packer that concatenates sequences.

    Sequences are concatenated until the target length is reached,
    then a new packed sequence is started.
    """

    def __init__(
        self,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        include_segment_ids: bool = True,
    ):
        """Initialize GreedyPacker.

        Args:
            seq_length: Target sequence length.
            eos_token_id: EOS token ID for separation.
            pad_token_id: Padding token ID.
            include_segment_ids: Whether to track segment IDs.
        """
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.include_segment_ids = include_segment_ids

        # Current buffer
        self._buffer: list[int] = []
        self._labels: list[int] = []
        self._segment_ids: list[int] = []
        self._current_segment = 1
        self._source_ids: list[str] = []

    def add(self, tokens: list[int], labels: list[int] | None = None, source_id: str | None = None) -> PackedSequence | None:
        """Add tokens to the packer.

        Args:
            tokens: Token IDs to add.
            labels: Optional labels aligned with tokens.
            source_id: Optional source identifier.

        Returns:
            PackedSequence if a full sequence is ready, None otherwise.
        """
        result = None

        tokens, labels = _normalize_sample(tokens, labels, self.eos_token_id, self.seq_length)

        # Add tokens to buffer
        if labels is not None and len(self._labels) < len(self._buffer):
            self._labels.extend([-100] * (len(self._buffer) - len(self._labels)))
        if labels is None and self._labels:
            labels = [-100] * len(tokens)

        for idx, tok in enumerate(tokens):
            self._buffer.append(tok)
            if labels is not None:
                self._labels.append(labels[idx])
            if self.include_segment_ids:
                self._segment_ids.append(self._current_segment)

            # Check if we have a full sequence
            if len(self._buffer) >= self.seq_length:
                result = self._flush()

        if tokens and source_id:
            self._source_ids.append(source_id)
        self._current_segment = max(self._segment_ids, default=0) + 1

        return result

    def _flush(self) -> PackedSequence:
        """Create a packed sequence from the current buffer."""
        # Take exactly seq_length tokens
        input_ids = np.array(self._buffer[: self.seq_length], dtype=np.int32)
        labels = np.array(self._labels[: self.seq_length], dtype=np.int32) if self._labels else None

        segment_ids = None
        if self.include_segment_ids:
            segment_ids = np.array(self._segment_ids[: self.seq_length], dtype=np.int32)

        result = PackedSequence(
            input_ids=input_ids,
            labels=labels,
            segment_ids=segment_ids,
            decoder_positions=_build_decoder_positions(segment_ids),
            source_ids=self._source_ids.copy() if self._source_ids else None,
            num_segments=max(self._segment_ids[: self.seq_length], default=0),
        )

        # Keep remainder
        self._buffer = self._buffer[self.seq_length :]
        if self._labels:
            self._labels = self._labels[self.seq_length :]
        if self.include_segment_ids:
            self._segment_ids = self._segment_ids[self.seq_length :]
        self._source_ids = []
        self._current_segment = max(self._segment_ids, default=0) + 1

        return result

    def flush_final(self) -> PackedSequence | None:
        """Flush any remaining tokens with padding."""
        if not self._buffer:
            return None

        # Never exceed seq_length, even if buffer is longer
        buf = self._buffer[: self.seq_length]
        label_buf = self._labels[: self.seq_length] if self._labels else []
        pad_len = self.seq_length - len(buf)

        input_ids = np.array(buf + [self.pad_token_id] * pad_len, dtype=np.int32)
        labels = np.array(label_buf + ([-100] * pad_len), dtype=np.int32) if self._labels else None

        attention_mask = np.zeros(self.seq_length, dtype=np.int32)
        attention_mask[: len(buf)] = 1

        seg: list[int] = []
        segment_ids = None
        if self.include_segment_ids:
            seg = self._segment_ids[: self.seq_length]
            padded_segments = seg + [_PADDING_SEGMENT_ID] * pad_len
            segment_ids = np.array(padded_segments, dtype=np.int32)

        result = PackedSequence(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            decoder_positions=_build_decoder_positions(segment_ids),
            source_ids=self._source_ids.copy() if self._source_ids else None,
            num_segments=max(seg, default=0),
        )

        self._buffer = []
        self._labels = []
        self._segment_ids = []
        self._source_ids = []
        self._current_segment = 1

        return result


class PoolPacker:
    """Pool of packers for more efficient bin-packing.

    Uses multiple packers to find better fits for sequences,
    reducing padding waste.
    """

    def __init__(
        self,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        num_packers: int = 4,
        include_segment_ids: bool = True,
    ):
        """Initialize PoolPacker.

        Args:
            seq_length: Target sequence length.
            eos_token_id: EOS token ID for separation.
            pad_token_id: Padding token ID.
            num_packers: Number of packers in the pool.
            include_segment_ids: Whether to track segment IDs.
        """
        self.seq_length = seq_length
        self.num_packers = num_packers
        self._packers = [
            GreedyPacker(seq_length, eos_token_id, pad_token_id, include_segment_ids) for _ in range(num_packers)
        ]

    def add(self, tokens: list[int], labels: list[int] | None = None, source_id: str | None = None) -> list[PackedSequence]:
        """Add tokens to the best-fit packer.

        Args:
            tokens: Token IDs to add.
            source_id: Optional source identifier.

        Returns:
            List of completed PackedSequences (may be empty).
        """
        results = []
        token_len = len(tokens)

        # Find the packer with the best fit (least remaining space after adding)
        best_idx = 0
        best_fit = float("inf")

        for i, packer in enumerate(self._packers):
            current_len = len(packer._buffer)
            estimated_tokens, _ = _normalize_sample(tokens, labels, packer.eos_token_id, self.seq_length)
            remaining_after = self.seq_length - (current_len + len(estimated_tokens))

            if 0 <= remaining_after < best_fit:
                best_fit = remaining_after
                best_idx = i

        # Add to best packer
        result = self._packers[best_idx].add(tokens, labels, source_id)
        if result is not None:
            results.append(result)

        return results

    def flush_all(self) -> list[PackedSequence]:
        """Flush all packers."""
        results = []
        for packer in self._packers:
            result = packer.flush_final()
            if result is not None:
                results.append(result)
        return results


class FirstFitPacker:
    """First-fit decreasing bin-packing packer.

    Collects sequences and uses first-fit decreasing algorithm
    for optimal packing efficiency.
    """

    def __init__(
        self,
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        include_segment_ids: bool = True,
        buffer_size: int = 1000,
    ):
        """Initialize FirstFitPacker.

        Args:
            seq_length: Target sequence length.
            eos_token_id: EOS token ID for separation.
            pad_token_id: Padding token ID.
            include_segment_ids: Whether to track segment IDs.
            buffer_size: Number of sequences to buffer before packing.
        """
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.include_segment_ids = include_segment_ids
        self.buffer_size = buffer_size

        self._pending: list[tuple[list[int], list[int] | None, str | None]] = []

    def add(self, tokens: list[int], labels: list[int] | None = None, source_id: str | None = None) -> list[PackedSequence]:
        """Add tokens to the pending buffer.

        Args:
            tokens: Token IDs to add.
            source_id: Optional source identifier.

        Returns:
            List of completed PackedSequences when buffer is full.
        """
        self._pending.append((tokens, labels, source_id))

        if len(self._pending) >= self.buffer_size:
            return self._pack_buffer()

        return []

    def _pack_buffer(self) -> list[PackedSequence]:
        """Pack the pending buffer using first-fit decreasing."""
        if not self._pending:
            return []

        normalized_pending = [
            (*_normalize_sample(raw_tokens, raw_labels, self.eos_token_id, self.seq_length), source_id)
            for raw_tokens, raw_labels, source_id in self._pending
        ]
        sorted_pending = sorted(normalized_pending, key=lambda x: len(x[0]), reverse=True)

        # Bins: list of [tokens, labels, segment_ids]
        bins: list[list[tp.Any]] = []

        for tokens, labels, source_id in sorted_pending:
            token_len = len(tokens)
            placed = False

            # Find first bin that fits
            for _i, bin_data in enumerate(bins):
                bin_tokens, bin_labels, bin_segments, bin_sources = bin_data
                if len(bin_tokens) + token_len <= self.seq_length:
                    # Add to this bin
                    segment_id = max(bin_segments) + 1 if bin_segments else 1
                    
                    if labels is not None and bin_labels is None:
                        bin_labels = [-100] * len(bin_tokens)
                        bin_data[1] = bin_labels
                        
                    bin_tokens.extend(tokens)
                    
                    if bin_labels is not None:
                        bin_labels.extend(labels if labels is not None else [-100] * len(tokens))
                    bin_segments.extend([segment_id] * len(tokens))
                    if source_id:
                        bin_sources.append(source_id)
                    placed = True
                    break

            if not placed:
                # Create new bin
                new_tokens = list(tokens)
                new_labels = list(labels) if labels is not None else None
                new_segments = [1] * len(new_tokens)
                new_sources = [source_id] if source_id else []
                bins.append([new_tokens, new_labels, new_segments, new_sources])

        # Convert bins to PackedSequences
        results = []
        for bin_tokens, bin_labels, bin_segments, bin_sources in bins:
            if len(bin_tokens) > self.seq_length:
                raise ValueError("packed bin exceeded seq_length after placement")

            # Pad if needed
            pad_len = self.seq_length - len(bin_tokens)
            input_ids = np.array(bin_tokens + [self.pad_token_id] * pad_len, dtype=np.int32)
            
            if bin_labels is not None:
                labels_arr = np.array(bin_labels + ([-100] * pad_len), dtype=np.int32)
            else:
                labels_arr = None

            attention_mask = np.ones(self.seq_length, dtype=np.int32)
            attention_mask[len(bin_tokens) :] = 0

            segment_ids = None
            if self.include_segment_ids:
                padded_segments = bin_segments + [_PADDING_SEGMENT_ID] * pad_len
                segment_ids = np.array(padded_segments, dtype=np.int32)

            results.append(
                PackedSequence(
                    input_ids=input_ids,
                    labels=labels_arr,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    decoder_positions=_build_decoder_positions(segment_ids),
                    source_ids=bin_sources if bin_sources else None,
                    num_segments=max(bin_segments, default=0),
                )
            )

        self._pending = []
        return results

    def flush_all(self) -> list[PackedSequence]:
        """Flush remaining pending sequences."""
        return self._pack_buffer()


class PackedShardedSource(ShardedDataSource[dict]):
    """Sharded source that packs sequences from another source."""

    def __init__(
        self,
        source: ShardedDataSource[dict],
        seq_length: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        strategy: str = "greedy",
        num_packers: int = 4,
        include_segment_ids: bool = True,
        input_field: str = "input_ids",
        shuffle: bool = True,
        shuffle_buffer_factor: int = 10,
        seed: int | None = None,
    ):
        """Initialize PackedShardedSource.

        Args:
            source: Source to pack.
            seq_length: Target sequence length.
            eos_token_id: EOS token ID.
            pad_token_id: Padding token ID.
            strategy: Packing strategy - "greedy", "pool", or "first_fit".
            num_packers: Number of packers for pool strategy.
            include_segment_ids: Whether to include segment IDs.
            input_field: Field name containing input IDs.
            shuffle: Whether to shuffle packed sequences.
            shuffle_buffer_factor: Buffer size multiplier for shuffling.
            seed: Random seed.
        """
        self._source = source
        self._seq_length = seq_length
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._strategy = strategy
        self._num_packers = num_packers
        self._include_segment_ids = include_segment_ids
        self._input_field = input_field
        self._shuffle = shuffle
        self._shuffle_buffer_factor = shuffle_buffer_factor
        self._seed = seed

    @property
    def shard_names(self) -> "Sequence[str]":
        return ["packed_shard_0"]

    def num_shards(self) -> int:
        return 1

    def _create_packer(self):
        """Create a packer based on strategy."""
        if self._strategy == "pool":
            return PoolPacker(
                self._seq_length,
                self._eos_token_id,
                self._pad_token_id,
                self._num_packers,
                self._include_segment_ids,
            )
        elif self._strategy == "first_fit":
            return FirstFitPacker(
                self._seq_length,
                self._eos_token_id,
                self._pad_token_id,
                self._include_segment_ids,
            )
        else:  # greedy
            return GreedyPacker(
                self._seq_length,
                self._eos_token_id,
                self._pad_token_id,
                self._include_segment_ids,
            )

    def open_shard(self, shard_name: str) -> "Iterator[dict]":
        """Open the packed shard."""
        if shard_name not in self.shard_names:
            raise KeyError(f"Unknown packed shard: {shard_name}")
        if self._seed is not None:
            random.seed(self._seed)

        packer = self._create_packer()
        shuffle_buffer = []
        max_buffer = self._shuffle_buffer_factor * 100  # Approximate batch size

        def emit(packed: PackedSequence):
            """Emit a packed sequence, handling shuffle."""
            result = packed.to_dict()
            if self._shuffle:
                if len(shuffle_buffer) < max_buffer:
                    shuffle_buffer.append(result)
                    return None
                else:
                    idx = random.randrange(0, max_buffer)
                    out = shuffle_buffer[idx]
                    shuffle_buffer[idx] = result
                    return out
            return result

        # Iterate through source
        for source_shard in self._source.shard_names:
            for example in self._source.open_shard(source_shard):
                tokens = example.get(self._input_field, [])
                if not tokens:
                    continue
                labels = example.get("labels")

                source_id = example.get("__source__")

                if isinstance(packer, (PoolPacker, FirstFitPacker)):
                    results = packer.add(list(tokens), list(labels) if labels is not None else None, source_id)
                    for packed in results:
                        out = emit(packed)
                        if out is not None:
                            yield out
                else:
                    result = packer.add(list(tokens), list(labels) if labels is not None else None, source_id)
                    if result is not None:
                        out = emit(result)
                        if out is not None:
                            yield out

        # Flush packer
        if isinstance(packer, (PoolPacker, FirstFitPacker)):
            for packed in packer.flush_all():
                out = emit(packed)
                if out is not None:
                    yield out
        else:
            final = packer.flush_final()
            if final is not None:
                out = emit(final)
                if out is not None:
                    yield out

        # Emit remaining shuffle buffer
        if self._shuffle:
            random.shuffle(shuffle_buffer)
            yield from shuffle_buffer

    def __len__(self) -> int:
        """Return estimated number of packed sequences.

        Note: This is an estimate based on source length. Actual count
        depends on token distribution and packing efficiency.

        Raises:
            TypeError: If the underlying source doesn't support len().
        """
        # Estimate based on average sequence length ratio
        # Assume ~70% packing efficiency as a rough heuristic
        source_len = len(self._source)
        # Rough estimate: each packed sequence contains ~1.4 original sequences on average
        return max(1, int(source_len / 1.4))

    def __repr__(self) -> str:
        return (
            f"PackedShardedSource(seq_length={self._seq_length}, strategy={self._strategy!r}, source={self._source!r})"
        )


class PackStage(BaseStage):
    """Pipeline stage for packing sequences."""

    def __init__(self, config: PackStageConfig | None = None):
        """Initialize PackStage.

        Args:
            config: Packing stage configuration.
        """
        super().__init__(config.__dict__ if config else {})
        self._stage_config = config or PackStageConfig()

    @property
    def name(self) -> str:
        return "pack"

    def process(
        self,
        data: dict[str, ShardedDataSource],
        context: PipelineContext,
    ) -> dict[str, ShardedDataSource]:
        """Pack sequences in all datasets.

        Args:
            data: Dictionary mapping dataset names to sources.
            context: Pipeline context.

        Returns:
            Dictionary with packed sources.
        """
        if not self._stage_config.enabled:
            return data

        result = {}
        for ds_name, source in data.items():
            packed = PackedShardedSource(
                source=source,
                seq_length=self._stage_config.seq_length,
                eos_token_id=self._stage_config.eos_token_id,
                pad_token_id=self._stage_config.pad_token_id,
                strategy=self._stage_config.strategy,
                num_packers=self._stage_config.num_packers,
                include_segment_ids=self._stage_config.include_segment_ids,
                shuffle=self._stage_config.shuffle_packed,
                shuffle_buffer_factor=self._stage_config.shuffle_buffer_factor,
                seed=context.seed,
            )
            result[ds_name] = packed
            logger.info(f"Packed dataset '{ds_name}' with strategy={self._stage_config.strategy}")

        return result


def pack_pre_tokenized(stream, seq_length: int, eos_token_id: int, batch_size: int, shuffle: bool, buffer_factor: int):
    """Pack pre-tokenized sequences into constant-length chunks.

    Takes a stream of pre-tokenized examples and packs them into fixed-length
    sequences for efficient training. Sequences are concatenated and split
    at the specified sequence length, with EOS tokens inserted as needed.

    Args:
        stream: Iterator of dictionaries containing 'tokens' field.
        seq_length: Target length for packed sequences.
        eos_token_id: Token ID to use for padding/separation.
        batch_size: Batch size (used for shuffle buffer calculation).
        shuffle: Whether to shuffle the packed sequences.
        buffer_factor: Multiplier for shuffle buffer size (batch_size * buffer_factor).

    Returns:
        Generator yielding dictionaries with 'input_ids' as JAX arrays.
    """

    def gen():
        buf = np.array([], dtype=np.int32)
        eos = np.array([eos_token_id], dtype=np.int32)
        shuffle_buf = []
        max_buf = batch_size * buffer_factor

        for sample in stream:
            toks = sample["tokens"]
            # Use asarray to avoid unnecessary copy if already int32 ndarray
            toks = np.asarray(toks, dtype=np.int32)
            buf = np.concatenate([buf, toks], axis=0)
            if len(buf) % seq_length != 0:
                buf = np.concatenate([buf, eos], axis=0)
            while len(buf) >= seq_length:
                ex = {"input_ids": jnp.array(buf[:seq_length])}
                buf = buf[seq_length:]
                if shuffle:
                    if len(shuffle_buf) < max_buf:
                        shuffle_buf.append(ex)
                    else:
                        i = random.randrange(0, max_buf)
                        yield shuffle_buf[i]
                        shuffle_buf[i] = ex
                else:
                    yield ex
        random.shuffle(shuffle_buf)
        for ex in shuffle_buf:
            yield ex

    return gen


def pack_constant_length(
    stream,
    tokenize_fn,
    seq_length: int,
    eos_token_id: int,
    batch_size: int,
    shuffle: bool,
    buffer_factor: int,
):
    """Pack sequences with on-the-fly tokenization into constant-length chunks.

    Combines tokenization and packing in a single pipeline. Takes raw examples,
    tokenizes them using the provided function, and packs the results into
    fixed-length sequences.

    Args:
        stream: Iterator of raw examples to tokenize.
        tokenize_fn: Function that takes an example and returns token IDs.
        seq_length: Target length for packed sequences.
        eos_token_id: Token ID to use for padding/separation.
        batch_size: Batch size (used for shuffle buffer calculation).
        shuffle: Whether to shuffle the packed sequences.
        buffer_factor: Multiplier for shuffle buffer size (batch_size * buffer_factor).

    Returns:
        Generator yielding dictionaries with 'input_ids' as JAX arrays.
    """

    def token_iter():
        for ex in stream:
            toks = tokenize_fn(ex)
            yield {"tokens": toks}

    return pack_pre_tokenized(token_iter(), seq_length, eos_token_id, batch_size, shuffle, buffer_factor)
