"""
sample_chain.py
---------------
A minimal walkthrough of Spotify's `pedalboard` library — the same kind of
effects chain you'd find on a guitar pedalboard, but in Python and fully
differentiable-friendly for MIR / audio-ML experiments.

Pipeline:
    input.wav  -->  HighpassFilter --> Compressor --> Chorus -->
                    Distortion --> Reverb --> Gain --> Limiter
                -->  output.wav

Run:
    python sample_chain.py input.wav output.wav

If you don't have audio handy, run `make_test_tone.py` first to generate one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pedalboard import (
    Pedalboard,
    Chorus,
    Compressor,
    Distortion,
    Gain,
    HighpassFilter,
    Limiter,
    Reverb,
)
from pedalboard.io import AudioFile


def build_board() -> Pedalboard:
    """Construct a small but representative effects chain.

    Order matters: dynamics (compressor) usually sit early so later stages
    react to a controlled signal level; time-based effects (reverb) usually
    sit late so they're applied to the already-shaped tone.
    """
    return Pedalboard(
        [
            # 1. Clean up rumble below 80 Hz before anything else touches it.
            HighpassFilter(cutoff_frequency_hz=80.0),

            # 2. Even out the dynamics. Threshold in dBFS, ratio is N:1.
            Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=5.0, release_ms=120.0),

            # 3. Subtle modulation — adds width and movement.
            Chorus(rate_hz=1.2, depth=0.25, mix=0.25),

            # 4. A touch of harmonic saturation. drive_db is gentle here.
            Distortion(drive_db=8.0),

            # 5. Space. room_size 0..1, damping rolls off the high tail.
            Reverb(room_size=0.35, damping=0.5, wet_level=1, dry_level=0.75),

            # 6. Make-up gain to compensate for the compressor pulling it down.
            Gain(gain_db=2.0),

            # 7. Catch any peaks before they clip the file on write.
            Limiter(threshold_db=-1.0, release_ms=100.0),
        ]
    )


def process(input_path: Path, output_path: Path, board: Pedalboard) -> None:
    """Stream the input file through the board in chunks and write the result.

    Streaming (vs. loading the whole file) keeps memory flat for long inputs,
    which matters when you start running this over a dataset.
    """
    with AudioFile(str(input_path)) as in_f:
        sample_rate = in_f.samplerate
        num_channels = in_f.num_channels

        with AudioFile(
            str(output_path),
            "w",
            samplerate=sample_rate,
            num_channels=num_channels,
        ) as out_f:
            buffer_size = 1 << 15  # 32k frames per chunk
            while in_f.tell() < in_f.frames:
                chunk = in_f.read(buffer_size)
                effected = board(chunk, sample_rate, reset=False)
                out_f.write(effected)

    print(f"Wrote {output_path}  (sr={sample_rate}, ch={num_channels})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to input .wav or .mp3")
    parser.add_argument("output", type=Path, help="Path to output file (.wav)")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Input file not found: {args.input}\n"
            f"Tip: run `python make_test_tone.py` to generate test_tone.wav."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    board = build_board()
    process(args.input, args.output, board)


if __name__ == "__main__":
    main()
