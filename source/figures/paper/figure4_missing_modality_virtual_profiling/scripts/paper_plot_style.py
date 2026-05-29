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
    "terracotta": "#C96144",
    "amber": "#E99D4E",
    "blue": "#5185C0",
    "violet": "#8281B9",
    "green": "#55966B",
    "mint": "#99C290",
    "lilac": "#C0BEDC",
    "sky": "#8EA9D4",
    "peach": "#F2CB9F",
    "slate": "#99AABB",
    "pink": "#FFB3C1",
    "blush": "#FFD3E0",
    "light_blue": "#87CEEB",
}

NEUTRAL_LINE = PALETTE["slate"]
NEUTRAL_LIGHT = PALETTE["lilac"]

AXIS_LABEL_FS = 13
TITLE_FS = 13
TICK_LABEL_FS = 10.5
LEGEND_FS = 10.5
PANEL_LABEL_FS = 16
SMALL_TEXT_FS = 11
SUPTITLE_FS = 14


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": TICK_LABEL_FS,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": AXIS_LABEL_FS,
            "axes.titlesize": TITLE_FS,
            "axes.titleweight": "bold",
            "xtick.labelsize": TICK_LABEL_FS,
            "ytick.labelsize": TICK_LABEL_FS,
            "legend.fontsize": LEGEND_FS,
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


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_FS,
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
