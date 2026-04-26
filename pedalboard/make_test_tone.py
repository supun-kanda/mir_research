"""
make_test_tone.py
-----------------
Generates a short stereo test signal so `sample_chain.py` has something to
process even before you bring in real audio.

The signal is an A-major chord arpeggio (A4, C#5, E5, A5) with a small
amplitude envelope on each note. It's musical enough that the effects
(reverb tail, chorus modulation, compressor pumping) become audibly obvious.

Run:
    python make_test_tone.py            # writes test_tone.wav
    python make_test_tone.py out.wav    # custom name
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pedalboard.io import AudioFile


SAMPLE_RATE = 44_100
NOTE_SECONDS = 0.6
NOTES_HZ = [440.0, 554.37, 659.25, 880.0]  # A4, C#5, E5, A5


def envelope(num_samples: int) -> np.ndarray:
    """Simple attack/decay envelope so notes don't click."""
    attack = int(0.02 * num_samples)
    decay = num_samples - attack
    env = np.concatenate(
        [
            np.linspace(0.0, 1.0, attack, dtype=np.float32),
            np.linspace(1.0, 0.0, decay, dtype=np.float32),
        ]
    )
    return env


def synth_note(freq_hz: float, seconds: float, sr: int) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    # mild harmonic content — fundamental + a touch of 2nd and 3rd harmonic
    wave = (
        0.7 * np.sin(2 * np.pi * freq_hz * t)
        + 0.2 * np.sin(2 * np.pi * 2 * freq_hz * t)
        + 0.1 * np.sin(2 * np.pi * 3 * freq_hz * t)
    )
    return (wave * envelope(n)).astype(np.float32)


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_tone.wav")

    mono = np.concatenate([synth_note(f, NOTE_SECONDS, SAMPLE_RATE) for f in NOTES_HZ])
    # Stereo: same signal both sides, slightly delayed on the right for width.
    delay = int(0.003 * SAMPLE_RATE)  # 3 ms
    right = np.concatenate([np.zeros(delay, dtype=np.float32), mono])[: len(mono)]
    stereo = np.stack([mono, right], axis=0)  # shape (channels, frames)

    # Headroom so the compressor in sample_chain.py has something to grab.
    stereo *= 0.5

    with AudioFile(str(out_path), "w", samplerate=SAMPLE_RATE, num_channels=2) as f:
        f.write(stereo)

    print(f"Wrote {out_path}  ({stereo.shape[1] / SAMPLE_RATE:.2f}s, stereo, {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
