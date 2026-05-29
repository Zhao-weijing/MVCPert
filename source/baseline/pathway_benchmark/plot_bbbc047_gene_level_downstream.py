#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_ORDER = ["baseline_multi", "dynamics_modulation", "mvcpert"]
MODEL_LABELS = {
    "baseline_multi": "Baseline Multi",
    "dynamics_modulation": "Dynamics_Modulation",
    "mvcpert": "MVCPert",
}
MODEL_COLORS = {
    "baseline_multi": "#4C78A8",
    "dynamics_modulation": "#F58518",
    "mvcpert": "#E45756",
}


def plot_top100_bar(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = summary_df.loc[(summary_df["subset"] == "informative") & (summary_df["top_k"] == 100)].copy()
    plot_df["model_key"] = pd.Categorical(plot_df["model_key"], categories=MODEL_ORDER, ordered=True)
    plot_df = plot_df.sort_values("model_key")
    labels = [MODEL_LABELS.get(k, k) for k in plot_df["model_key"].astype(str).tolist()]
    overlap_values = plot_df["mean_overlap_ratio"].tolist()
    sign_values = plot_df["mean_sign_recall"].tolist()
    colors = [MODEL_COLORS.get(k, "#9C9C9C") for k in plot_df["model_key"].astype(str).tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(labels, overlap_values, color=colors)
    axes[0].set_title("Informative: Top100 Overlap")
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axes[0].set_axisbelow(True)

    axes[1].bar(labels, sign_values, color=colors)
    axes[1].set_title("Informative: Top100 Sign Recall")
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    out = output_dir / "bbbc047_gene_level_top100_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_top100_sample_boxplot(sample_df: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = sample_df.loc[(sample_df["subset"] == "informative") & (sample_df["top_k"] == 100)].copy()
    labels = [MODEL_LABELS[k] for k in MODEL_ORDER]
    overlap_data = [plot_df.loc[plot_df["model_key"] == k, "overlap_ratio"].tolist() for k in MODEL_ORDER]
    sign_data = [plot_df.loc[plot_df["model_key"] == k, "sign_recall"].tolist() for k in MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    b1 = axes[0].boxplot(overlap_data, tick_labels=labels, patch_artist=True)
    b2 = axes[1].boxplot(sign_data, tick_labels=labels, patch_artist=True)
    for patch, model_key in zip(b1["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model_key])
        patch.set_alpha(0.7)
    for patch, model_key in zip(b2["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model_key])
        patch.set_alpha(0.7)
    axes[0].set_title("Sample-level Top100 Overlap")
    axes[0].tick_params(axis="x", rotation=12)
    axes[1].set_title("Sample-level Top100 Sign Recall")
    axes[1].tick_params(axis="x", rotation=12)
    for ax in axes:
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    out = output_dir / "bbbc047_gene_level_top100_sample_boxplot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    summary_df = pd.read_csv(OUTPUT_DIR / "bbbc047_gene_level_downstream_summary.csv")
    sample_df = pd.read_csv(OUTPUT_DIR / "bbbc047_gene_level_downstream_sample_metrics.csv")
    output_paths = [
        plot_top100_bar(summary_df, OUTPUT_DIR),
        plot_top100_sample_boxplot(sample_df, OUTPUT_DIR),
    ]
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
