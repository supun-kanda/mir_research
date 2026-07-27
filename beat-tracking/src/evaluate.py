"""Run both beat trackers over ASAP/MAESTRO performances, score them against
ground truth with mir_eval, and attach per-piece rubato measures.

This is the step that turns "leaderboard" into "finding": every row carries
both an F-measure per tracker and a rubato measure, so F-measure can later
be plotted as a function of rubato (see scripts/plot_results.py).
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Iterable, Optional

import mir_eval
import pandas as pd
from tqdm import tqdm

from . import datasets, rubato
from .trackers import TRACKERS

# mir_eval.beat.f_measure default threshold is already 0.07s == +/-70ms,
# matching the ASAP evaluation convention described in thesis.md.
F_MEASURE_THRESHOLD = 0.07


def evaluate_one(performance: dict) -> dict:
    """Run every tracker on one performance and score it. `performance` is
    one item as yielded by datasets.iter_performances_with_audio."""
    reference_beats = datasets.load_beat_times(performance["annotation_path"])

    row = {
        "composer": performance["composer"],
        "title": performance["title"],
        "folder": performance["folder"],
        "audio_path": str(performance["audio_path"]),
    }
    row.update(rubato.rubato_measures(reference_beats))

    for name, track_fn in TRACKERS.items():
        try:
            estimated_beats = track_fn(performance["audio_path"])
            f_measure = mir_eval.beat.f_measure(
                reference_beats, estimated_beats, f_measure_threshold=F_MEASURE_THRESHOLD
            )
        except Exception:
            f_measure = float("nan")
            row[f"{name}_error"] = traceback.format_exc(limit=1)
        row[f"{name}_f_measure"] = f_measure

    return row


def run_experiment(
    asap_dir: Path,
    output_csv: Path,
    unique_pieces: bool = True,
    max_pieces: Optional[int] = None,
    performances: Optional[Iterable[dict]] = None,
) -> pd.DataFrame:
    """Evaluate every available performance and write results to CSV.

    Only performances with a built audio file (see
    scripts/build_asap_audio.py) and an annotation file are used.
    """
    if performances is None:
        performances = list(
            datasets.iter_performances_with_audio(asap_dir, unique_pieces=unique_pieces)
        )
    else:
        performances = list(performances)

    if not performances:
        raise RuntimeError(
            "No performances with built audio found. Run scripts/download_asap.sh "
            "and scripts/build_asap_audio.py first."
        )

    if max_pieces is not None:
        performances = performances[:max_pieces]

    rows = [evaluate_one(p) for p in tqdm(performances, desc="Evaluating performances")]
    df = pd.DataFrame(rows)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
