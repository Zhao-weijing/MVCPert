#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure4_latent_geometry_helper import build_joint_panel_payloads, draw_joint_panel, latent_branch_handles
from paper_plot_style import (
    AXIS_LABEL_FS,
    FIG_DIR,
    LEGEND_FS,
    NEUTRAL_LINE,
    PALETTE,
    SMALL_TEXT_FS,
    SUPTITLE_FS,
    TICK_LABEL_FS,
    TITLE_FS,
    add_panel_label,
    apply_publication_style,
    clean_axes,
    with_alpha,
)


ARTIFACT_ROOT = Path(os.environ.get("MVCPERT_ARTIFACT_ROOT", "<ARTIFACT_ROOT>"))
MAIN_RUN_REL = (
    "BBBC047/official_v1/MVCPert_HyperGateResidualVAE/2026-05-13/"
    "MVC_HyperGateResidualVAE_hypergate_residual_vae_three_view_corrected_main_e40_BBBC047_smiles_split/"
    "ECFP4_Default/predict"
)

PROFILE_H5_PATH = Path(
    os.environ.get(
        "MVCPERT_FIG4_PROFILE_H5",
        str(ARTIFACT_ROOT / MAIN_RUN_REL / "test_ps_all_views_prediction_profile.h5"),
    )
)
LATENT_PATH = Path(
    os.environ.get(
        "MVCPERT_FIG4_LATENTS_NPZ",
        str(ARTIFACT_ROOT / MAIN_RUN_REL / "test_ps_latents.npz"),
    )
)
MULTISEED_ROOT = Path(
    os.environ.get(
        "MVCPERT_FIG4_MULTISEED_ROOT",
        str(ARTIFACT_ROOT)
        + "/BBBC047/followups/E43_bbbc047_hypergate_residual_vae_three_view_multiseed/"
        "formal_train/20260516_223900/train_outputs/2026-05-16",
    )
)
MULTISEED_RUNS = [
    PROFILE_H5_PATH,
    MULTISEED_ROOT
    / "MVC_HyperGateResidualVAE_hypergate_residual_vae_three_view_corrected_main_e40_seed0_BBBC047_smiles_split/ECFP4_Default/predict/test_ps_all_views_prediction_profile.h5",
    MULTISEED_ROOT
    / "MVC_HyperGateResidualVAE_hypergate_residual_vae_three_view_corrected_main_e40_seed10_BBBC047_smiles_split/ECFP4_Default/predict/test_ps_all_views_prediction_profile.h5",
    MULTISEED_ROOT
    / "MVC_HyperGateResidualVAE_hypergate_residual_vae_three_view_corrected_main_e40_seed100_BBBC047_smiles_split/ECFP4_Default/predict/test_ps_all_views_prediction_profile.h5",
    MULTISEED_ROOT
    / "MVC_HyperGateResidualVAE_hypergate_residual_vae_three_view_corrected_main_e40_seed1000_BBBC047_smiles_split/ECFP4_Default/predict/test_ps_all_views_prediction_profile.h5",
]

SCHEMATIC_PDF_PATH = FIG_DIR / "figure4_missing_modality_training_schematic.pdf"
OUTPUT_STEM = "figure4_bbbc047_missing_modality_virtual_profiling"

COLOR_FULL = PALETTE["blue"]
COLOR_MISSING_CP = PALETTE["amber"]
COLOR_MISSING_GE = PALETTE["green"]
COLOR_NEUTRAL = NEUTRAL_LINE


def branch_center_shift_2d(reference: np.ndarray, current: np.ndarray) -> float:
    return float(np.linalg.norm(current.mean(axis=0) - reference.mean(axis=0)))


def gaussian_overlap_2d(reference: np.ndarray, current: np.ndarray, grid_size: int = 160) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)

    ref_mean = ref.mean(axis=0)
    cur_mean = cur.mean(axis=0)
    ref_cov = np.cov(ref.T) + np.eye(2) * 1e-4
    cur_cov = np.cov(cur.T) + np.eye(2) * 1e-4

    combined = np.vstack([ref, cur])
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    mins = mins - 0.15 * spans
    maxs = maxs + 0.15 * spans

    x = np.linspace(mins[0], maxs[0], grid_size)
    y = np.linspace(mins[1], maxs[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    pos = np.stack([xx, yy], axis=-1)

    def gaussian_pdf(mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        det = max(det, 1e-12)
        delta = pos - mean
        exponent = np.einsum("...i,ij,...j->...", delta, inv, delta)
        norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
        return norm * np.exp(-0.5 * exponent)

    ref_pdf = gaussian_pdf(ref_mean, ref_cov)
    cur_pdf = gaussian_pdf(cur_mean, cur_cov)
    dx = float(x[1] - x[0]) if grid_size > 1 else 1.0
    dy = float(y[1] - y[0]) if grid_size > 1 else 1.0
    overlap = np.minimum(ref_pdf, cur_pdf).sum() * dx * dy
    return float(np.clip(overlap, 0.0, 1.0))


def save_outputs(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ("pdf",):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300)
        written.append(path)
    return written


def load_h5_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {key: handle[key][()] for key in handle.keys()}


def rowwise_pcc(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.astype(np.float64, copy=False)
    pred = pred.astype(np.float64, copy=False)
    true_centered = true - true.mean(axis=1, keepdims=True)
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(true_centered, axis=1) * np.linalg.norm(pred_centered, axis=1)
    return np.divide(
        np.sum(true_centered * pred_centered, axis=1),
        denom,
        out=np.zeros(true.shape[0], dtype=np.float64),
        where=denom > 1e-12,
    )


def rowwise_top_abs_overlap(true_delta: np.ndarray, pred_delta: np.ndarray, k: int = 100) -> np.ndarray:
    k = min(k, true_delta.shape[1])
    true_idx = np.argpartition(np.abs(true_delta), -k, axis=1)[:, -k:]
    pred_idx = np.argpartition(np.abs(pred_delta), -k, axis=1)[:, -k:]
    values = []
    for true_row, pred_row in zip(true_idx, pred_idx):
        values.append(len(set(true_row.tolist()) & set(pred_row.tolist())) / k)
    return np.asarray(values, dtype=np.float64)


def rowwise_sign_consistency(true_delta: np.ndarray, pred_delta: np.ndarray, k: int = 100) -> np.ndarray:
    k = min(k, true_delta.shape[1])
    idx = np.argpartition(np.abs(true_delta), -k, axis=1)[:, -k:]
    rows = np.arange(true_delta.shape[0])[:, None]
    return np.mean(np.sign(true_delta[rows, idx]) == np.sign(pred_delta[rows, idx]), axis=1)


def branch_arrays(arrays: dict[str, np.ndarray], setting: str, branch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true = arrays[f"target_{branch}"]
    pred = arrays[f"{setting}_{branch}_pred"]
    control = arrays[f"control_{branch}"]
    return true, pred, control


def compute_dataset_metrics(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for setting in ("full", "geview", "cpview"):
        output[setting] = {}
        for branch in ("cp", "ge"):
            true, pred, control = branch_arrays(arrays, setting, branch)
            true_delta = true - control
            pred_delta = pred - control
            output[setting][branch] = {
                "profile_pcc": float(np.mean(rowwise_pcc(true, pred))),
                "delta_pcc": float(np.mean(rowwise_pcc(true_delta, pred_delta))),
                "top100_overlap": float(np.mean(rowwise_top_abs_overlap(true_delta, pred_delta, k=100))),
                "sign100": float(np.mean(rowwise_sign_consistency(true_delta, pred_delta, k=100))),
            }
    return output


def compute_latent_stability(latent_path: Path) -> dict[str, dict[str, float]]:
    data = np.load(latent_path)
    result: dict[str, dict[str, float]] = {}
    for setting in ("geview", "cpview"):
        result[setting] = {}
        for branch in ("shared", "ge", "cp"):
            full = np.asarray(data[f"full_z_{branch}"], dtype=np.float64)
            masked = np.asarray(data[f"{setting}_z_{branch}"], dtype=np.float64)
            denom = np.linalg.norm(full, axis=1) * np.linalg.norm(masked, axis=1)
            cos = np.divide(
                np.sum(full * masked, axis=1),
                denom,
                out=np.zeros(full.shape[0], dtype=np.float64),
                where=denom > 1e-12,
            )
            result[setting][branch] = float(cos.mean())
    return result


def summarize_latent_retention(
    ordered_payloads: list[dict[str, object]],
    latent_stability: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    payload_by_prefix = {str(payload["prefix"]): payload for payload in ordered_payloads}
    full_payload = payload_by_prefix["full"]
    summaries: dict[str, dict[str, float]] = {}
    for prefix, stability_key in (("cpview", "cpview"), ("geview", "geview")):
        payload = payload_by_prefix[prefix]
        branch_cosines = latent_stability[stability_key]
        center_shifts = {
            "shared": branch_center_shift_2d(full_payload["dir_shared"], payload["dir_shared"]),
            "ge": branch_center_shift_2d(full_payload["dir_ge"], payload["dir_ge"]),
            "cp": branch_center_shift_2d(full_payload["dir_cp"], payload["dir_cp"]),
        }
        overlap_scores = {
            "shared": gaussian_overlap_2d(full_payload["dir_shared"], payload["dir_shared"]),
            "ge": gaussian_overlap_2d(full_payload["dir_ge"], payload["dir_ge"]),
            "cp": gaussian_overlap_2d(full_payload["dir_cp"], payload["dir_cp"]),
        }
        summaries[prefix] = {
            "mean_cosine": float(np.mean([branch_cosines["shared"], branch_cosines["ge"], branch_cosines["cp"]])),
            "mean_overlap_2d": float(np.mean([overlap_scores["shared"], overlap_scores["ge"], overlap_scores["cp"]])),
            "mean_center_shift_2d": float(np.mean([center_shifts["shared"], center_shifts["ge"], center_shifts["cp"]])),
            "shared_cosine": float(branch_cosines["shared"]),
            "ge_cosine": float(branch_cosines["ge"]),
            "cp_cosine": float(branch_cosines["cp"]),
            "shared_overlap_2d": float(overlap_scores["shared"]),
            "ge_overlap_2d": float(overlap_scores["ge"]),
            "cp_overlap_2d": float(overlap_scores["cp"]),
            "shared_center_shift_2d": float(center_shifts["shared"]),
            "ge_center_shift_2d": float(center_shifts["ge"]),
            "cp_center_shift_2d": float(center_shifts["cp"]),
        }
    return summaries


def annotate_retention_metrics(ax: plt.Axes, summary: dict[str, float], title: str) -> None:
    lines = [
        title,
        f"Mean overlap to D: {summary['mean_overlap_2d']:.3f}",
        f"Mean center shift: {summary['mean_center_shift_2d']:.3f}",
    ]
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SMALL_TEXT_FS,
        fontweight="bold",
        color="#304255",
        bbox=dict(
            boxstyle="round,pad=0.32",
            facecolor=with_alpha("#F7F8FB", 0.97),
            edgecolor="#7E8FA6",
            linewidth=1.0,
        ),
    )


def load_pdf_preview_image(pdf_path: Path, dpi: int = 220) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="figure4_panel_a_") as tmpdir:
        prefix = Path(tmpdir) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return plt.imread(prefix.with_suffix(".png"))


def draw_task_schematic_panel(ax: plt.Axes, pdf_path: Path) -> None:
    image = load_pdf_preview_image(pdf_path)
    ax.imshow(image)
    ax.set_title("Missing-Modality Training Protocol", fontsize=TITLE_FS)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def compute_paper_metric_bundle(arrays: dict[str, np.ndarray], pred_prefix: str, branch: str) -> dict[str, float]:
    true = np.asarray(arrays[f"target_{branch}"], dtype=np.float64)
    pred = np.asarray(arrays[f"{pred_prefix}_{branch}_pred"], dtype=np.float64)
    control = np.asarray(arrays[f"control_{branch}"], dtype=np.float64)
    target_centroid = true.mean(axis=0, keepdims=True)
    return {
        "PCC": float(np.mean(rowwise_pcc(true, pred))),
        "system-PCC": float(np.mean(rowwise_pcc(true - target_centroid, pred - target_centroid))),
        "DEG-PCC": float(np.mean(rowwise_pcc(true - control, pred - control))),
    }


def load_multiseed_panel_metrics() -> dict[str, dict[str, list[dict[str, float]]]]:
    panel_metrics = {
        "CP -> GE": {"Full input": [], "CP-only": []},
        "GE -> CP": {"Full input": [], "GE-only": []},
    }
    for path in MULTISEED_RUNS:
        arrays = load_h5_arrays(path)
        full_ge = compute_paper_metric_bundle(arrays, "full", "ge")
        cp_only_ge = compute_paper_metric_bundle(arrays, "cpview", "ge")
        full_cp = compute_paper_metric_bundle(arrays, "full", "cp")
        ge_only_cp = compute_paper_metric_bundle(arrays, "geview", "cp")
        panel_metrics["CP -> GE"]["Full input"].append(full_ge)
        panel_metrics["CP -> GE"]["CP-only"].append(cp_only_ge)
        panel_metrics["GE -> CP"]["Full input"].append(full_cp)
        panel_metrics["GE -> CP"]["GE-only"].append(ge_only_cp)
    return panel_metrics


def draw_multiseed_bar_panel(
    ax: plt.Axes,
    records: dict[str, list[dict[str, float]]],
    title: str,
    only_label: str,
    only_color: str,
    metric_labels: list[str],
) -> None:
    metric_names = ["PCC", "system-PCC", "DEG-PCC"]
    x = np.arange(len(metric_names), dtype=np.float64)
    width = 0.30

    full_means = np.asarray([np.mean([row[name] for row in records["Full input"]]) for name in metric_names], dtype=np.float64)
    only_means = np.asarray([np.mean([row[name] for row in records[only_label]]) for name in metric_names], dtype=np.float64)
    full_stds = np.asarray([np.std([row[name] for row in records["Full input"]], ddof=1) for name in metric_names], dtype=np.float64)
    only_stds = np.asarray([np.std([row[name] for row in records[only_label]], ddof=1) for name in metric_names], dtype=np.float64)

    ax.bar(
        x - width / 2,
        full_means,
        width=width,
        color=with_alpha(COLOR_FULL, 0.82),
        edgecolor="white",
        linewidth=0.8,
        yerr=full_stds,
        ecolor="#586579",
        capsize=3,
        label="Full input",
    )
    ax.bar(
        x + width / 2,
        only_means,
        width=width,
        color=with_alpha(only_color, 0.78),
        edgecolor="white",
        linewidth=0.8,
        yerr=only_stds,
        ecolor="#586579",
        capsize=3,
        label=only_label,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 0.80)
    ax.set_ylabel("Mean ± s.d.", fontsize=AXIS_LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.grid(False)
    ax.tick_params(axis="x", length=0, labelsize=TICK_LABEL_FS)
    ax.tick_params(axis="y", length=3.2, width=0.8, colors="#4D5968", labelsize=TICK_LABEL_FS)
    ax.spines["left"].set_color("#4D5968")
    ax.spines["bottom"].set_color("#4D5968")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4, borderaxespad=0.2, fontsize=LEGEND_FS)
    clean_axes(ax)


def generate_figure() -> list[Path]:
    arrays = load_h5_arrays(PROFILE_H5_PATH)
    dataset_metrics = compute_dataset_metrics(arrays)
    latent_stability = compute_latent_stability(LATENT_PATH)
    joint_payloads = build_joint_panel_payloads(dict(np.load(LATENT_PATH)))
    multiseed_metrics = load_multiseed_panel_metrics()

    apply_publication_style()
    fig = plt.figure(figsize=(15.8, 12.2))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 0.86],
        wspace=0.60,
        hspace=0.48,
    )

    ax_a = fig.add_subplot(gs[0, 1:5])
    draw_task_schematic_panel(ax_a, SCHEMATIC_PDF_PATH)
    add_panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[1, 0:3])
    draw_multiseed_bar_panel(
        ax_b,
        multiseed_metrics["CP -> GE"],
        "CP-to-GE Completion Performance",
        "CP-only",
        COLOR_MISSING_GE,
        ["GE-PCC", "GE system-PCC", "GE-DEG-PCC"],
    )
    add_panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 3:6])
    draw_multiseed_bar_panel(
        ax_c,
        multiseed_metrics["GE -> CP"],
        "GE-to-CP Completion Performance",
        "GE-only",
        COLOR_MISSING_CP,
        ["CP-PCC", "CP system-PCC", "CP-DEG-PCC"],
    )
    add_panel_label(ax_c, "C")

    retention = {
        "CP profile": dataset_metrics["geview"]["cp"]["profile_pcc"] / dataset_metrics["full"]["cp"]["profile_pcc"],
        "CP delta": dataset_metrics["geview"]["cp"]["delta_pcc"] / dataset_metrics["full"]["cp"]["delta_pcc"],
        "GE profile": dataset_metrics["cpview"]["ge"]["profile_pcc"] / dataset_metrics["full"]["ge"]["profile_pcc"],
        "GE delta": dataset_metrics["cpview"]["ge"]["delta_pcc"] / dataset_metrics["full"]["ge"]["delta_pcc"],
    }

    bottom_titles = [
        "Full-input Joint Latent Geometry",
        "CP-only Input Joint Latent Geometry",
        "GE-only Input Joint Latent Geometry",
    ]
    ordered_payloads = [joint_payloads[0], joint_payloads[2], joint_payloads[1]]
    latent_retention_summary = summarize_latent_retention(ordered_payloads, latent_stability)
    x_values = np.concatenate(
        [
            payload["dir_shared"][:, 0] for payload in ordered_payloads
        ]
        + [
            payload["dir_ge"][:, 0] for payload in ordered_payloads
        ]
        + [
            payload["dir_cp"][:, 0] for payload in ordered_payloads
        ]
    )
    y_values = np.concatenate(
        [
            payload["dir_shared"][:, 1] for payload in ordered_payloads
        ]
        + [
            payload["dir_ge"][:, 1] for payload in ordered_payloads
        ]
        + [
            payload["dir_cp"][:, 1] for payload in ordered_payloads
        ]
    )
    x_pad = 0.05 * max(float(x_values.max() - x_values.min()), 1e-6)
    y_pad = 0.05 * max(float(y_values.max() - y_values.min()), 1e-6)
    shared_xlim = (float(x_values.min() - x_pad), float(x_values.max() + x_pad))
    shared_ylim = (float(y_values.min() - y_pad), float(y_values.max() + y_pad))

    bottom_slices = [(0, 2), (2, 4), (4, 6)]
    for (start, end), (payload, title, label) in zip(bottom_slices, zip(ordered_payloads, bottom_titles, ["D", "E", "F"])):
        ax = fig.add_subplot(gs[2, start:end])
        draw_joint_panel(
            ax,
            payload["dir_shared"],
            payload["dir_ge"],
            payload["dir_cp"],
            title,
            xlim=shared_xlim,
            ylim=shared_ylim,
        )
        add_panel_label(ax, label)
        if payload["prefix"] == "cpview":
            annotate_retention_metrics(ax, latent_retention_summary["cpview"], "Retention vs full input")
        elif payload["prefix"] == "geview":
            annotate_retention_metrics(ax, latent_retention_summary["geview"], "Retention vs full input")

    fig.legend(
        handles=latent_branch_handles(),
        frameon=True,
        facecolor=with_alpha("#F7F8FB", 0.98),
        edgecolor="#7E8FA6",
        framealpha=1.0,
        fontsize=LEGEND_FS,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.6,
        handletextpad=0.5,
        borderpad=0.55,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.97, bottom=0.10)
    written = save_outputs(fig, OUTPUT_STEM)
    plt.close(fig)
    return written


def main() -> int:
    outputs = generate_figure()
    print("Saved Figure 4 assets:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
