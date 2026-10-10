"""The decision-drift dataset is valid and split 10/10 (bench/dataset.py)."""

import copy

import pytest

from bench import dataset


def test_dataset_is_valid_and_split_evenly() -> None:
    projects = dataset.load()
    assert len(projects) == 20
    assert [p["split"] for p in projects].count("calibration") == 10


def test_an_outdated_answer_is_rejected() -> None:
    p = copy.deepcopy(dataset.load()[0])
    p["questions"][0]["answer"] = "e1"  # e1 is superseded in shopfront
    with pytest.raises(ValueError, match="outdated"):
        dataset.check(p)
