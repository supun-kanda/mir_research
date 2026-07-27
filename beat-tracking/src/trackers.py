"""Thin wrappers around the two beat trackers under comparison.

librosa.beat.beat_track: onset-strength envelope + Ellis-style dynamic
programming against a single global tempo estimate -- penalizes departure
from that one tempo, so it should be structurally hostile to rubato.

madmom's RNN + DBN pipeline: a recurrent net produces a beat-activation
function, decoded by a dynamic Bayesian network / HMM that allows tempo to
drift within learned transition probabilities -- rubato-tolerant by
construction.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import librosa
import numpy as np


def track_beats_librosa(
    audio_path: Union[str, Path], sr: int = 22050
) -> np.ndarray:
    """Beat times (seconds) from librosa's onset-envelope + DP beat tracker."""
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    _, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return np.asarray(beat_times, dtype=float)


@lru_cache(maxsize=1)
def _madmom_processors():
    # Instantiating these loads the RNN model weights, so cache module-wide
    # rather than re-loading per file.
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

    return RNNBeatProcessor(), DBNBeatTrackingProcessor(fps=100)


def track_beats_madmom(audio_path: Union[str, Path]) -> np.ndarray:
    """Beat times (seconds) from madmom's RNN activation + DBN decoder."""
    activation_proc, dbn_proc = _madmom_processors()
    activations = activation_proc(str(audio_path))
    beat_times = dbn_proc(activations)
    return np.asarray(beat_times, dtype=float)


TRACKERS = {
    "librosa": track_beats_librosa,
    "madmom": track_beats_madmom,
}
