"""Load the raw Starbucks data, either from ``data.zip`` or from an
extracted directory.

The data set has three files:

- ``portfolio.json`` — one row per offer (10 offers).
- ``profile.json`` — one row per customer (17,000 customers).
- ``transcript.json`` — one row per event (306,534 events).

All three are JSON-lines files. Reading straight from the zip keeps the
working tree clean: the extracted JSON files (~42 MB) never need to exist
on disk.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd

from . import config

_FILES = {
    "portfolio": "portfolio.json",
    "profile": "profile.json",
    "transcript": "transcript.json",
}


def load_from_zip(zip_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw tables from a zip archive.

    The archive is expected to contain the files under a ``data/``
    prefix (as in ``data.zip``); a flat layout is accepted too.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"data archive not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        frames = {}
        for key, filename in _FILES.items():
            member = f"data/{filename}" if f"data/{filename}" in names else filename
            if member not in names:
                raise KeyError(f"{filename} not found in {zip_path}")
            with zf.open(member) as fh:
                frames[key] = pd.read_json(fh, orient="records", lines=True)
    return frames["portfolio"], frames["profile"], frames["transcript"]


def load_from_dir(dir_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw tables from an extracted directory.

    Accepts either the directory that directly contains the ``*.json``
    files or its parent (i.e. a directory containing a ``data/``
    subdirectory).
    """
    dir_path = Path(dir_path)
    data_dir = dir_path / "data" if (dir_path / "data").is_dir() else dir_path
    frames = {}
    for key, filename in _FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"expected data file not found: {path}")
        frames[key] = pd.read_json(path, orient="records", lines=True)
    return frames["portfolio"], frames["profile"], frames["transcript"]


def load_raw(
    source: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the raw tables from a zip archive or a directory.

    Defaults to ``data.zip`` at the repository root.
    """
    source = Path(source) if source is not None else config.DEFAULT_DATA_SOURCE
    if source.is_dir():
        return load_from_dir(source)
    return load_from_zip(source)


def sample_transcript(
    transcript: pd.DataFrame,
    fraction: float,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Return a random row subsample of the transcript.

    Intended for smoke runs only (e.g. CI): subsampling events breaks the
    pairing between an offer being received and the same offer being
    completed, so metrics from a subsample are not comparable to a full run.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return transcript
    return transcript.sample(frac=fraction, random_state=random_state).reset_index(drop=True)
