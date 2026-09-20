#!/usr/bin/env python3
"""Plot tracker F-measure against a per-piece rubato measure -- the
plot that turns the librosa-vs-madmom comparison into a finding (see
thesis.md): does F-measure fall off a cliff as rubato increases, or bend?

Usage:
    python scripts/plot_results.py --results results/results.csv --rubato-measure ibi_std
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Two-series categorical palette (validated for CVD + contrast; see the
# dataviz skill's references/palette.md -- slots 1 and 2 in fixed order).
COLOR_LIBROSA = "#2a78d6"  # categorical slot 1, blue
COLOR_MADMOM = "#eb6834"   # categorical slot 2, orange
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

TRACKER_COLORS = {"librosa": COLOR_LIBROSA, "madmom": COLOR_MADMOM}


def binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 8):
    """Median F-measure within equal-count bins of the rubato measure --
    a robust stand-in for a smoothed trend line without extra deps."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.array_split(np.arange(len(x)), n_bins)
    xs, ys = [], []
    for idx in edges:
        if len(idx) == 0:
            continue
        xs.append(np.median(x[idx]))
        ys.append(np.median(y[idx]))
    return np.array(xs), np.array(ys)


def plot(df: pd.DataFrame, rubato_measure: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for tracker, color in TRACKER_COLORS.items():
        col = f"{tracker}_f_measure"
        if col not in df:
            continue
        sub = df.dropna(subset=[rubato_measure, col])
        x = sub[rubato_measure].to_numpy()
        y = sub[col].to_numpy()

        ax.scatter(x, y, s=22, color=color, alpha=0.45, linewidths=0, label=None)

        if len(x) >= 4:
            bx, by = binned_trend(x, y)
            ax.plot(bx, by, color=color, linewidth=2.5, label=tracker)
        else:
            ax.scatter([], [], color=color, label=tracker)  # legend entry only

    ax.set_xlabel(f"Rubato measure ({rubato_measure})", color=INK)
    ax.set_ylabel(f"F-measure (mir_eval.beat, +/-70ms)", color=INK)
    ax.set_title("Beat-tracking F-measure vs. rubato", color=INK, fontsize=13)
    ax.set_ylim(-0.02, 1.02)

    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED)

    legend = ax.legend(frameon=False, loc="lower left")
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/results.csv"))
    parser.add_argument(
        "--rubato-measure", default="ibi_std",
        choices=["ibi_std", "ibi_cv", "local_tempo_wobble"],
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    output_path = args.output or args.results.parent / f"f_measure_vs_{args.rubato_measure}.png"
    plot(df, args.rubato_measure, output_path)


if __name__ == "__main__":
    main()
