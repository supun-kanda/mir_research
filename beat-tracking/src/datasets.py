"""Access to the ASAP dataset (scores + beat annotations) and its matched
MAESTRO audio subset.

ASAP ships annotations, scores and MIDI performances directly in its git
repo; audio is *not* included (copyright) and has to be cut out of the
separately-downloaded MAESTRO dataset using the `start`/`end` columns in
metadata.csv. See scripts/download_asap.sh and scripts/build_asap_audio.py.

Reference: https://github.com/fosfrancesco/asap-dataset (README + metadata.csv,
checked against the live repo while writing this module).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

# Annotation TSV labels that mark a beat (see ASAP README, "TSV annotation"):
#   b  -> ordinary beat
#   db -> downbeat (also a beat)
#   bR -> beat whose exact position can't be established from notation
#         (rubato passages, mid-score pickup measures) but is still usable
#         for beat tracking per the dataset authors.
BEAT_LABELS = {"b", "db", "bR"}
DOWNBEAT_LABELS = {"db"}


def load_metadata(asap_dir: Path) -> pd.DataFrame:
    """Load metadata.csv from a cloned ASAP dataset directory."""
    asap_dir = Path(asap_dir)
    path = asap_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Clone the ASAP dataset first "
            "(see scripts/download_asap.sh)."
        )
    return pd.read_csv(path)


def _parse_annotation_tsv(path: Path) -> pd.DataFrame:
    """Parse an ASAP `*_annotations.txt` TSV file.

    Each line is `time\\ttime\\tlabel`, where label may itself be several
    comma-separated sub-labels when multiple event types coincide at the
    same timestamp (e.g. "db,4/4" for a downbeat that's also a time
    signature change). See the ASAP README usage example, which resolves
    this by splitting on comma and looking at the first token.
    """
    df = pd.read_csv(
        path, header=None, names=["time", "time2", "type"], sep="\t", dtype=str
    )
    df["primary_label"] = df["type"].fillna("").apply(lambda s: s.split(",")[0])
    df["time"] = df["time"].astype(float)
    return df


def load_beat_times(annotation_path: Path) -> np.ndarray:
    """Sorted, de-duplicated beat times (b + db + bR) from an annotation TSV."""
    df = _parse_annotation_tsv(annotation_path)
    beats = df.loc[df["primary_label"].isin(BEAT_LABELS), "time"]
    return np.sort(beats.unique())


def load_downbeat_times(annotation_path: Path) -> np.ndarray:
    """Sorted, de-duplicated downbeat times from an annotation TSV."""
    df = _parse_annotation_tsv(annotation_path)
    beats = df.loc[df["primary_label"].isin(DOWNBEAT_LABELS), "time"]
    return np.sort(beats.unique())


def resolve_audio_path(asap_dir: Path, row: pd.Series) -> Optional[Path]:
    """Path to the (already-cut) audio performance for a metadata row, or
    None if this performance has no audio or the file hasn't been built yet
    (see scripts/build_asap_audio.py)."""
    audio_performance = row.get("audio_performance")
    if not isinstance(audio_performance, str) or not audio_performance:
        return None
    path = Path(asap_dir) / audio_performance
    return path if path.exists() else None


def resolve_annotation_path(asap_dir: Path, row: pd.Series) -> Optional[Path]:
    annotations = row.get("performance_annotations")
    if not isinstance(annotations, str) or not annotations:
        return None
    path = Path(asap_dir) / annotations
    return path if path.exists() else None


def iter_performances_with_audio(
    asap_dir: Path, unique_pieces: bool = False
) -> Iterator[dict]:
    """Yield dicts for every ASAP performance that currently has both a
    built audio file on disk and a beat annotation file.

    Set unique_pieces=True to keep only one performance per (composer,
    title) pair -- useful so pieces with many recorded performances don't
    dominate the evaluation.
    """
    asap_dir = Path(asap_dir)
    df = load_metadata(asap_dir)
    if unique_pieces:
        df = df.drop_duplicates(subset=["title", "composer"])

    for _, row in df.iterrows():
        audio_path = resolve_audio_path(asap_dir, row)
        annotation_path = resolve_annotation_path(asap_dir, row)
        if audio_path is None or annotation_path is None:
            continue
        yield {
            "composer": row["composer"],
            "title": row["title"],
            "folder": row["folder"],
            "audio_path": audio_path,
            "annotation_path": annotation_path,
        }


def count_available_audio(asap_dir: Path) -> int:
    """How many performances already have a built audio file -- a quick way
    to check whether scripts/build_asap_audio.py has been run."""
    return sum(1 for _ in iter_performances_with_audio(asap_dir))
