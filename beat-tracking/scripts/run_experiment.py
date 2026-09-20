#!/usr/bin/env python3
"""Run the librosa vs madmom beat-tracking comparison over ASAP/MAESTRO and
write per-performance results (F-measure + rubato measures) to CSV.

Requires the ASAP dataset cloned and audio built first:
    ./scripts/download_asap.sh
    ./scripts/download_maestro.sh
    python scripts/build_asap_audio.py --maestro data/maestro-v2.0.0/maestro-v2.0.0

Usage:
    python scripts/run_experiment.py --asap data/asap-dataset --output results/results.csv
    python scripts/run_experiment.py --max-pieces 20   # quick partial run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap", type=Path, default=Path("data/asap-dataset"))
    parser.add_argument("--output", type=Path, default=Path("results/results.csv"))
    parser.add_argument(
        "--all-performances", action="store_true",
        help="Evaluate every performance, not just one per unique piece.",
    )
    parser.add_argument("--max-pieces", type=int, default=None)
    args = parser.parse_args()

    df = run_experiment(
        asap_dir=args.asap,
        output_csv=args.output,
        unique_pieces=not args.all_performances,
        max_pieces=args.max_pieces,
    )

    print(f"\nWrote {len(df)} rows to {args.output}")
    for tracker in ("librosa", "madmom"):
        col = f"{tracker}_f_measure"
        if col in df:
            print(f"  {tracker}: mean F-measure = {df[col].mean():.3f}")


if __name__ == "__main__":
    main()
