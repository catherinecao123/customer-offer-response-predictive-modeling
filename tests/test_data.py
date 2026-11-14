"""Tests for offer_response.data."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from offer_response import data


def _write_json_lines(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows))


@pytest.fixture
def tiny_dataset(tmp_path):
    """A minimal on-disk data set (one row per file)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_json_lines(
        data_dir / "portfolio.json",
        [{"id": "o1", "offer_type": "bogo", "difficulty": 5, "reward": 5,
          "duration": 7, "channels": ["email"]}],
    )
    _write_json_lines(
        data_dir / "profile.json",
        [{"id": "c1", "age": 40, "became_member_on": 20180101,
          "gender": "F", "income": 50000.0}],
    )
    _write_json_lines(
        data_dir / "transcript.json",
        [{"event": "offer received", "person": "c1", "time": 0,
          "value": {"offer id": "o1"}}],
    )
    return tmp_path


def test_load_from_dir(tiny_dataset):
    portfolio, profile, transcript = data.load_from_dir(tiny_dataset)
    assert len(portfolio) == len(profile) == len(transcript) == 1
    assert portfolio.iloc[0]["offer_type"] == "bogo"


def test_load_from_dir_accepts_data_subdir(tiny_dataset):
    portfolio, _profile, _transcript = data.load_from_dir(tiny_dataset / "data")
    assert len(portfolio) == 1


def test_load_from_zip(tiny_dataset, tmp_path):
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in ("portfolio.json", "profile.json", "transcript.json"):
            zf.write(tiny_dataset / "data" / name, f"data/{name}")
    portfolio, profile, transcript = data.load_from_zip(zip_path)
    assert len(portfolio) == len(profile) == len(transcript) == 1


def test_load_raw_dispatches_on_path_type(tiny_dataset):
    _portfolio, _profile, transcript = data.load_raw(tiny_dataset)
    assert transcript.iloc[0]["event"] == "offer received"
    with pytest.raises(FileNotFoundError):
        data.load_raw(tiny_dataset / "missing.zip")


def test_sample_transcript_is_deterministic():
    frame = pd.DataFrame({"event": ["x"] * 100, "n": range(100)})
    first = data.sample_transcript(frame, 0.1)
    second = data.sample_transcript(frame, 0.1)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 10


def test_sample_transcript_validates_fraction():
    frame = pd.DataFrame({"event": ["x"]})
    with pytest.raises(ValueError):
        data.sample_transcript(frame, 0)
    with pytest.raises(ValueError):
        data.sample_transcript(frame, 1.5)
