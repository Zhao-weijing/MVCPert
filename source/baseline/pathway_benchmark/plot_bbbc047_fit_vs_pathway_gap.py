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


def plot_fit_vs_pathway_scatter(sample_df: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = sample_df.loc[(sample_df["is_informative"] == 1)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for model_key in MODEL_ORDER:
        sub = plot_df.loc[plot_df["model_key"] == model_key].copy()
        if sub.empty:
            continue
        ax.scatter(
            pd.to_numeric(sub["all_gene_pcc"], errors="coerce"),
            pd.to_numeric(sub["pathway_jaccard"], errors="coerce"),
            alpha=0.8,
            s=38,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
        )
    ax.set_xlabel("All-gene PCC (informative subset)")
    ax.set_ylabel("Pathway Jaccard")
    ax.set_title("Fit vs Downstream Pathway Recovery")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = output_dir / "bbbc047_fit_vs_pathway_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_conservativeness_distribution(sample_df: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = sample_df.loc[(sample_df["is_informative"] == 1)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    ratio_data = [
        pd.to_numeric(plot_df.loc[plot_df["model_key"] == m, "pred_gt_l2_ratio"], errors="coerce").dropna().to_numpy()
        for m in MODEL_ORDER
    ]
    strong_data = [
        pd.to_numeric(plot_df.loc[plot_df["model_key"] == m, "pred_strong_gene_count"], errors="coerce").dropna().to_numpy()
        for m in MODEL_ORDER
    ]
    b1 = axes[0].boxplot(ratio_data, tick_labels=labels, patch_artist=True)
    b2 = axes[1].boxplot(strong_data, tick_labels=labels, patch_artist=True)
    for patch, model_key in zip(b1["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model_key])
        patch.set_alpha(0.7)
    for patch, model_key in zip(b2["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model_key])
        patch.set_alpha(0.7)
    axes[0].set_title("Pred/GT L2 ratio")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    axes[1].set_title("Strong-signal gene count")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    for ax in axes:
        ax.set_axisbelow(True)
    fig.tight_layout()
    out = output_dir / "bbbc047_conservativeness_distribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_topk_sensitivity_heatmap(sensitivity_df: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = sensitivity_df.loc[sensitivity_df["model_key"].isin(MODEL_ORDER)].copy()
    pivot = (
        plot_df.pivot(index="model_key", columns="top_k_genes", values="mean_pathway_jaccard")
        .reindex(index=MODEL_ORDER)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns.tolist()])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([MODEL_LABELS.get(k, k) for k in pivot.index.tolist()])
    ax.set_xlabel("Top-k genes (ORA)")
    ax.set_title("Top-k Sensitivity (informative subset mean pathway jaccard)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean pathway jaccard")
    fig.tight_layout()
    out = output_dir / "bbbc047_topk_sensitivity_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    sample_df = pd.read_csv(OUTPUT_DIR / "bbbc047_fit_pathway_gap_sample_metrics.csv")
    sensitivity_df = pd.read_csv(OUTPUT_DIR / "bbbc047_fit_pathway_gap_topk_sensitivity.csv")
    output_paths = [
        plot_fit_vs_pathway_scatter(sample_df, OUTPUT_DIR),
        plot_conservativeness_distribution(sample_df, OUTPUT_DIR),
        plot_topk_sensitivity_heatmap(sensitivity_df, OUTPUT_DIR),
    ]
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
