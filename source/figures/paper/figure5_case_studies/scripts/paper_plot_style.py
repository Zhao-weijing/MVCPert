#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


DPI = 300
FIGURE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = FIGURE_ROOT / "outputs"
ANALYSIS_DIR = FIGURE_ROOT / "analysis"


PALETTE = {
    "baseline": "#5F6F8C",
    "gatedpcc": "#2E8B80",
    "hypergate": "#D07A2D",
    "reference": "#A7ADB4",
    "tradeoff": "#7E6E68",
    "ground_truth": "#2E2E2E",
    "ge": "#3F7E48",
    "cp": "#B85C38",
    "dual": "#324E7B",
    "positive_fill": "#E6F1E8",
    "negative_fill": "#F5E6DE",
    "neutral_fill": "#F3EFE6",
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#D9D1C7",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "mathtext.fontset": "stix",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def with_alpha(color: str, alpha: float) -> tuple[float, float, float, float]:
    rgb = list(mcolors.to_rgb(color))
    return (rgb[0], rgb[1], rgb[2], alpha)


def blend(color: str, target: str = "#FFFFFF", fraction: float = 0.65) -> tuple[float, float, float]:
    base = mcolors.to_rgb(color)
    ref = mcolors.to_rgb(target)
    return tuple((1.0 - fraction) * base[i] + fraction * ref[i] for i in range(3))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ("pdf",):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path)
        written.append(path)
    return written
