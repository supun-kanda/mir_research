#!/usr/bin/env python3
"""Cut MAESTRO audio down to the ASAP performance clips, using the
start/end columns in metadata.csv, and copy them into the ASAP folder tree
at the `audio_performance` path.

This reimplements ASAP's own initialize_dataset.py, which can no longer run
as-is because it calls `librosa.output.write_wav`, removed from librosa
since 0.8 (we're on librosa 0.11). Critically, it also reproduces that
script's front-padding behaviour: when a clip's `start` > 0, 0.5s of
silence is prepended rather than starting the clip at time 0. The
ASAP beat annotations were generated against audio built this way, so
skipping the padding would leave every annotated beat time off by ~0.5s
for any performance with start > 0 (most of them).

Usage:
    python scripts/build_asap_audio.py --maestro data/maestro-v2.0.0/maestro-v2.0.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

PADDING_SECONDS = 0.5


def clip_and_copy_audio(in_path: Path, out_path: Path, start: float, end: float) -> None:
    if pd.isna(start) and pd.isna(end):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(Path(in_path).read_bytes())
        return

    s = 0.0 if pd.isna(start) else float(start)
    dur = None if pd.isna(end) else float(end) - s

    y, sr = librosa.load(str(in_path), sr=None, mono=False, offset=s, duration=dur)

    if s > 0:
        pad_samples = int(sr * PADDING_SECONDS)
        if y.ndim == 1:
            y = np.concatenate([np.zeros(pad_samples, dtype=y.dtype), y])
        else:
            zeros = np.zeros((y.shape[0], pad_samples), dtype=y.dtype)
            y = np.concatenate([zeros, y], axis=1)

    # soundfile wants (frames,) or (frames, channels); librosa gives
    # (channels, frames) for multi-channel.
    data = y.T if y.ndim == 2 else y

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PCM16, not float: madmom's built-in WAV reader can't handle IEEE-float
    # WAV and falls back to needing ffmpeg for anything that isn't native
    # int PCM (see scripts/smoke_test.py's comment on the same issue).
    sf.write(str(out_path), data.astype(np.float32), sr, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asap", type=Path, default=Path("data/asap-dataset"),
        help="Path to the cloned ASAP dataset repo.",
    )
    parser.add_argument(
        "--maestro", type=Path, required=True,
        help="Path to the extracted MAESTRO v2.0.0 directory (contains year "
        "subfolders like 2004/, 2006/, ...).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only build the first N clips (for a quick smoke test).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip clips whose output file already exists (default: on).",
    )
    args = parser.parse_args()

    metadata = pd.read_csv(args.asap / "metadata.csv")
    rows = metadata[metadata["maestro_audio_performance"].notna()]
    if args.limit is not None:
        rows = rows.head(args.limit)

    n_ok, n_skip, n_fail = 0, 0, 0
    for _, row in tqdm(list(rows.iterrows()), desc="Building ASAP audio clips"):
        out_path = args.asap / row["audio_performance"]
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue

        maestro_path = Path(
            str(row["maestro_audio_performance"]).replace("{maestro}", str(args.maestro))
        )
        if not maestro_path.exists():
            print(f"Missing MAESTRO source file, skipping: {maestro_path}")
            n_fail += 1
            continue

        try:
            clip_and_copy_audio(maestro_path, out_path, row["start"], row["end"])
            n_ok += 1
        except Exception as e:
            print(f"Failed for {row['audio_performance']}: {e}")
            n_fail += 1

    print(f"\nBuilt {n_ok}, skipped {n_skip} (already existed), failed {n_fail}.")


if __name__ == "__main__":
    main()
