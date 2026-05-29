#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_plot_style import (
    AXIS_LABEL_FS,
    LEGEND_FS,
    NEUTRAL_LINE,
    PANEL_LABEL_FS,
    PALETTE,
    SMALL_TEXT_FS,
    TITLE_FS,
    TICK_LABEL_FS,
    clean_axes,
    with_alpha,
)


RANDOM_STATE = 3407
COLOR_SHARED_FULL = PALETTE["blue"]
COLOR_GE_PRIVATE = PALETTE["green"]
COLOR_CP_PRIVATE = PALETTE["terracotta"]
COLOR_NEUTRAL = NEUTRAL_LINE


def build_direction_only_joint_latent(
    shared: np.ndarray,
    ge_private: np.ndarray,
    cp_private: np.ndarray,
) -> tuple[np.ndarray, int]:
    private_dim = ge_private.shape[1]
    if cp_private.shape[1] != private_dim:
        raise ValueError("GE-private and CP-private must have the same dimensionality.")

    shared_base = normalize(shared, norm="l2")
    ge_base = normalize(ge_private, norm="l2")
    cp_base = normalize(cp_private, norm="l2")

    if shared.shape[1] != private_dim:
        shared_reduce = PCA(n_components=private_dim, random_state=RANDOM_STATE)
        shared_base = shared_reduce.fit_transform(shared_base)

    shared_mix = normalize(shared_base, norm="l2")
    joint = np.concatenate([shared_mix, ge_base, cp_base], axis=0)
    return joint, shared.shape[0]


def split_joint_projection(joint_2d: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shared_2d = joint_2d[:n]
    ge_2d = joint_2d[n : 2 * n]
    cp_2d = joint_2d[2 * n :]
    return shared_2d, ge_2d, cp_2d


def build_joint_panel_payloads(latents: dict[str, np.ndarray]) -> list[dict[str, object]]:
    view_specs = [
        ("full", "Full input"),
        ("geview", "GE-only / MaskCP"),
        ("cpview", "CP-only / MaskGE"),
    ]
    joint_inputs: dict[str, tuple[np.ndarray, int]] = {}
    for prefix, _ in view_specs:
        shared = np.asarray(latents[f"{prefix}_z_shared"], dtype=np.float32)
        ge_private = np.asarray(latents[f"{prefix}_z_ge"], dtype=np.float32)
        cp_private = np.asarray(latents[f"{prefix}_z_cp"], dtype=np.float32)
        joint_inputs[prefix] = build_direction_only_joint_latent(shared, ge_private, cp_private)

    # Fit one reference PCA on full-view latents, then transform GE-only and CP-only
    # into the same 2D basis so D/E/F become directly comparable.
    full_joint, _ = joint_inputs["full"]
    reference_pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reference_pca.fit(full_joint)

    payloads: list[dict[str, object]] = []
    for prefix, title in view_specs:
        joint, n = joint_inputs[prefix]
        joint_2d = reference_pca.transform(joint)
        dir_shared, dir_ge, dir_cp = split_joint_projection(joint_2d, n)
        payloads.append(
            {
                "prefix": prefix,
                "title": title,
                "dir_shared": dir_shared,
                "dir_ge": dir_ge,
                "dir_cp": dir_cp,
                "dir_var": reference_pca.explained_variance_ratio_,
            }
        )
    return payloads


def draw_joint_panel(
    ax: plt.Axes,
    z_shared: np.ndarray,
    z_ge: np.ndarray,
    z_cp: np.ndarray,
    title: str,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    ax.scatter(z_shared[:, 0], z_shared[:, 1], s=9, c=COLOR_SHARED_FULL, alpha=0.17, linewidths=0, marker="o")
    ax.scatter(z_ge[:, 0], z_ge[:, 1], s=10, c=COLOR_GE_PRIVATE, alpha=0.22, linewidths=0, marker="s")
    ax.scatter(z_cp[:, 0], z_cp[:, 1], s=10, c=COLOR_CP_PRIVATE, alpha=0.22, linewidths=0, marker="^")
    shared_center = z_shared.mean(axis=0)
    ge_center = z_ge.mean(axis=0)
    cp_center = z_cp.mean(axis=0)
    ax.scatter([shared_center[0]], [shared_center[1]], s=110, c=COLOR_SHARED_FULL, marker="o", edgecolors="white", linewidths=0.9)
    ax.scatter([ge_center[0]], [ge_center[1]], s=110, c=COLOR_GE_PRIVATE, marker="s", edgecolors="white", linewidths=0.9)
    ax.scatter([cp_center[0]], [cp_center[1]], s=120, c=COLOR_CP_PRIVATE, marker="^", edgecolors="white", linewidths=0.9)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel("PC1", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("PC2", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FS, colors="#4D5968")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    clean_axes(ax)


def latent_branch_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_SHARED_FULL, markeredgecolor="#4D5968", markeredgewidth=0.7, markersize=7.8, label="Shared"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_GE_PRIVATE, markeredgecolor="#4D5968", markeredgewidth=0.7, markersize=7.8, label="GE-private"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_CP_PRIVATE, markeredgecolor="#4D5968", markeredgewidth=0.7, markersize=8.4, label="CP-private"),
    ]
