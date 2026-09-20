#!/usr/bin/env python3
"""End-to-end pipeline smoke test using synthetic click tracks

Generates a handful of synthetic "performances" at increasing amounts of
programmed rubato (a base tempo with a sinusoidal push/pull applied to the
beat spacing), renders each to a short click-track WAV, runs both trackers,
scores them with mir_eval, and reports F-measure vs. the rubato measure --
i.e. it exercises exactly the same code path scripts/run_experiment.py and
scripts/plot_results.py use on real data, so you can confirm the environment
and pipeline work before spending time on the real (large) dataset download.

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mir_eval

from src import rubato
from src.trackers import TRACKERS

SR = 22050
BASE_BPM = 100.0
N_BEATS = 120
RUBATO_LEVELS = {
    "none": 0.0,
    "mild": 0.08,
    "heavy": 0.30,
}


def make_click_track(rubato_depth: float, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Beat times with a sinusoidal tempo wobble, and the rendered click-track
    audio for them. rubato_depth is the fractional swing in inter-beat
    interval (0 = perfectly steady tempo)."""
    base_ibi = 60.0 / BASE_BPM
    beat_idx = np.arange(N_BEATS)
    # Smooth, slow (4-beat-period) modulation, not per-beat jitter -- meant to
    # emulate phrase-level push/pull rather than noise.
    wobble = 1.0 + rubato_depth * np.sin(2 * np.pi * beat_idx / 16.0)
    ibi = base_ibi * wobble
    beat_times = np.concatenate([[0.5], 0.5 + np.cumsum(ibi)])

    duration = beat_times[-1] + 1.0
    audio = np.zeros(int(duration * sr))
    click_dur = 0.03
    t = np.arange(int(click_dur * sr)) / sr
    click = np.sin(2 * np.pi * 1200 * t) * np.exp(-t / 0.01)

    for bt in beat_times:
        start = int(bt * sr)
        end = min(start + len(click), len(audio))
        audio[start:end] += click[: end - start]

    audio = 0.8 * audio / (np.max(np.abs(audio)) + 1e-9)
    # madmom's built-in WAV reader only handles integer PCM, not IEEE float
    # (it falls back to needing ffmpeg otherwise) -- write 16-bit PCM.
    return beat_times, audio.astype(np.float32)


def main() -> None:
    out_dir = Path("results/smoke_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'level':<8} {'rubato_ibi_std':>15} {'librosa_F':>10} {'madmom_F':>10}")
    for level, depth in RUBATO_LEVELS.items():
        beat_times, audio = make_click_track(depth)
        wav_path = out_dir / f"click_{level}.wav"
        sf.write(str(wav_path), audio, SR, subtype="PCM_16")

        measures = rubato.rubato_measures(beat_times)

        scores = {}
        for name, track_fn in TRACKERS.items():
            estimated = track_fn(wav_path)
            scores[name] = mir_eval.beat.f_measure(beat_times, estimated)

        print(
            f"{level:<8} {measures['ibi_std']:>15.4f} "
            f"{scores['librosa']:>10.3f} {scores['madmom']:>10.3f}"
        )

    print(f"\nClick-track WAVs written to {out_dir}/ -- pipeline ran end to end.")


if __name__ == "__main__":
    main()
