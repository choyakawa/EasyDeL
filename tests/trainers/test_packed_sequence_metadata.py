import pytest

from easydel.trainers.prompt_utils import add_packed_sequence_metadata


def test_add_packed_sequence_metadata_resets_positions_and_segments():
    example = {
        "input_ids": [11, 12, 13, 21, 22],
        "seq_lengths": [3, 2],
        "attention_mask": [1, 1, 1, 1, 1],
    }

    output = add_packed_sequence_metadata(example)

    assert output["position_ids"] == [0, 1, 2, 0, 1]
    assert output["segment_ids"] == [1, 1, 1, 2, 2]
    assert output["attention_mask"] == [1, 1, 1, 1, 1]


def test_add_packed_sequence_metadata_creates_attention_mask_when_missing():
    example = {
        "input_ids": [11, 12, 21],
        "seq_lengths": [2, 1],
    }

    output = add_packed_sequence_metadata(example)

    assert output["attention_mask"] == [1, 1, 1]
    assert output["position_ids"] == [0, 1, 0]
    assert output["segment_ids"] == [1, 1, 2]


def test_add_packed_sequence_metadata_requires_seq_lengths():
    with pytest.raises(ValueError):
        add_packed_sequence_metadata({"input_ids": [1, 2, 3]})
