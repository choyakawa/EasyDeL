from __future__ import annotations

from types import SimpleNamespace

from easydel.data.sources.base import HuggingFaceShardedSource


def test_streaming_huggingface_source_exposes_split_num_examples_metadata():
    source = HuggingFaceShardedSource(
        dataset_name="dummy/dataset",
        split="train",
        streaming=True,
    )
    source._dataset = SimpleNamespace(
        info=SimpleNamespace(
            splits={
                "train": SimpleNamespace(num_examples=123),
            }
        )
    )

    info = source.get_shard_info(source.shard_names[0])

    assert info is not None
    assert info.num_rows == 123
