"""Per-piece rubato measures computed from ground-truth beat times.

These quantify "how much does tempo wobble" so that tracker F-measure can be
plotted against it (the whole point of the experiment -- see thesis.md).
"""
from __future__ import annotations

import numpy as np


def inter_beat_intervals(beat_times: np.ndarray) -> np.ndarray:
    beat_times = np.asarray(beat_times, dtype=float)
    return np.diff(beat_times)


def ibi_std(beat_times: np.ndarray) -> float:
    """Standard deviation of inter-beat intervals, in seconds.

    The most direct reading of "rubato" from thesis.md: how much the gap
    between consecutive beats varies over the piece. Confounded with
    absolute tempo (a slow piece has bigger IBIs and so more room for
    seconds-scale variation) -- see ibi_cv for a tempo-normalized version.
    """
    ibi = inter_beat_intervals(beat_times)
    if ibi.size < 2:
        return float("nan")
    return float(np.std(ibi))


def ibi_cv(beat_times: np.ndarray) -> float:
    """Coefficient of variation of IBIs (std / mean): relative tempo
    fluctuation, comparable across pieces at different absolute tempi."""
    ibi = inter_beat_intervals(beat_times)
    if ibi.size < 2 or np.mean(ibi) == 0:
        return float("nan")
    return float(np.std(ibi) / np.mean(ibi))


def local_tempo_wobble(beat_times: np.ndarray, window: int = 9) -> float:
    """Std of instantaneous tempo (BPM) after removing a slow-moving trend
    (median filter over `window` beats).

    Isolates beat-to-beat push/pull from smooth accelerando/ritardando
    arcs or genuine long-term tempo shifts, which arguably aren't "rubato"
    in the thesis's sense of expressive push-and-pull.
    """
    ibi = inter_beat_intervals(beat_times)
    if ibi.size < window + 1:
        return float("nan")
    tempo = 60.0 / ibi
    half = window // 2
    padded = np.pad(tempo, (half, half), mode="edge")
    trend = np.array(
        [np.median(padded[i : i + window]) for i in range(len(tempo))]
    )
    residual = tempo - trend
    return float(np.std(residual))


def rubato_measures(beat_times: np.ndarray) -> dict:
    return {
        "n_beats": len(beat_times),
        "duration": float(beat_times[-1] - beat_times[0]) if len(beat_times) else 0.0,
        "ibi_std": ibi_std(beat_times),
        "ibi_cv": ibi_cv(beat_times),
        "local_tempo_wobble": local_tempo_wobble(beat_times),
    }
