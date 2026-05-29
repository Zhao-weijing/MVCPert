#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import shutil
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
FIGURE_ROOT = SCRIPT_DIR.parents[1] / 'figure3_architecture_ablation'
OUTPUT_DIR = FIGURE_ROOT / 'outputs'
LOCAL_ANALYSIS_DIR = FIGURE_ROOT / 'analysis'
SHARED_DIR = SCRIPT_DIR.parents[1] / 'shared'
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from paper_plot_style import add_panel_label, clean_axes
from paper_figure_data_common import (
    BBBC047_PROFILES,
    E40_MODAL_ABLATION_ROOT,
    METHOD_COLORS,
    STRUCTURE_PCC_METRICS,
    init_plot_env as _shared_init_plot_env,
)

METRIC_LABELS = {
    'systema-GE PCC': 'GE system',
    'GE PCC': 'GE profile',
    'DEG-GE PCC': 'GE-DEG',
    'systema-CP PCC': 'CP system',
    'CP PCC': 'CP profile',
    'DEG-CP PCC': 'CP-DEG',
}
AUGMENT_MODEL_ORDER = ['Baseline', 'HyperGate', 'MVCPert']
SEED_ORDER = [3407, 0, 10, 100, 1000]
FIG2_PALETTE = {
    'terracotta': '#C96144',
    'amber': '#E99D4E',
    'blue': '#5185C0',
    'violet': '#8281B9',
    'green': '#55966B',
    'mint': '#99C290',
    'lilac': '#C0BEDC',
    'sky': '#8EA9D4',
    'peach': '#F2CB9F',
    'slate': '#99AABB',
    'pink': '#FFB3C1',
    'blush': '#FFD3E0',
    'light_blue': '#87CEEB',
}
LEGACY_PRIMARY_NAME = 'Trans' + 'Morph'
LEGACY_DUAL_NAME = 'Dual ' + LEGACY_PRIMARY_NAME
LEGACY_HYPERGATE_NAME = 'TM-' + 'HyperGate'
LEGACY_LONG_HYPERGATE_NAME = LEGACY_PRIMARY_NAME + '-HyperGate'
LEGACY_TM_PREFIX = 'TM' + ' minus '
LEGACY_TM_HG_PREFIX = 'TM-' + 'HG minus '
LEGACY_LABEL_MAP = {
    LEGACY_PRIMARY_NAME: 'MVCPert',
    LEGACY_DUAL_NAME: 'Dual MVCPert',
    LEGACY_HYPERGATE_NAME: 'MVCPert-HyperGate',
    LEGACY_LONG_HYPERGATE_NAME: 'MVCPert-HyperGate',
    LEGACY_TM_PREFIX + 'Baseline': 'MVCPert minus MVCPert-Base',
    LEGACY_TM_PREFIX + 'best GE-only': 'MVCPert minus best GE-only',
    LEGACY_TM_PREFIX + 'best literature': 'MVCPert minus best literature',
    LEGACY_TM_HG_PREFIX + 'Baseline': 'MVCPert-HG minus MVCPert-Base',
}
FIG2_MODEL_COLORS = {
    'Baseline': FIG2_PALETTE['terracotta'],
    'Gate': FIG2_PALETTE['amber'],
    'HyperGate': FIG2_PALETTE['green'],
    'MVCPert': FIG2_PALETTE['blue'],
}
DISPLAY_MODEL_LABELS = {
    'Baseline': 'MVCPert-Base',
    'HyperGate': 'MVCPert-HyperGate',
    'MVCPert': 'MVCPert',
}
PANEL_D_LEGEND_LABELS = {
    'Baseline': 'Base',
    'HyperGate': 'HyperGate',
    'MVCPert': 'MVCPert',
}
FIG2_NEUTRAL_LINE = FIG2_PALETTE['slate']
FIG2_NEUTRAL_LIGHT = FIG2_PALETTE['lilac']
ARTIFACT_ROOT = Path(os.environ.get("MVCPERT_ARTIFACT_ROOT", "<ARTIFACT_ROOT>"))
STRUCTURE_MULTISEED_ROOT = Path(os.environ.get(
    "MVCPERT_FIG3_STRUCTURE_MULTISEED_ROOT",
    str(ARTIFACT_ROOT / "BBBC047/followups/E41_bbbc047_structure_multiseed_ci/formal_train"),
))
STRUCTURE_CI_TAGS = {
    'Baseline': {
        3407: BBBC047_PROFILES['Baseline'],
        0: ('MVC', 'e41_bbbc047_baseline_augbest40_seed0'),
        10: ('MVC', 'e41_bbbc047_baseline_augbest40_seed10'),
        100: ('MVC', 'e41_bbbc047_baseline_augbest40_seed100'),
        1000: ('MVC', 'e41_bbbc047_baseline_augbest40_seed1000'),
    },
    'Gate': {
        3407: ARTIFACT_ROOT / 'BBBC047/followups/E26_bbbc047_baseline_to_hypergate_7run_ablation/formal_train/20260421_debug/train_outputs/2026-04-21/MVC_GatedPCC_e26_bbbc047_baseline_to_hypergate_7run_gatedpcc_augbest40_v1_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        0: ('MVC_GatedPCC', 'e43_bbbc047_gatedpcc_augbest40_seed0'),
        10: ('MVC_GatedPCC', 'e43_bbbc047_gatedpcc_augbest40_seed10'),
        100: ('MVC_GatedPCC', 'e43_bbbc047_gatedpcc_augbest40_seed100'),
        1000: ('MVC_GatedPCC', 'e43_bbbc047_gatedpcc_augbest40_seed1000'),
    },
    'HyperGate': {
        3407: ARTIFACT_ROOT / 'BBBC047/followups/E26_bbbc047_baseline_to_hypergate_7run_ablation/formal_train/20260421_debug/train_outputs/2026-04-21/MVC_HyperGate_e26_bbbc047_baseline_to_hypergate_7run_hypergate_augbest40_v1_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        0: ('MVC_HyperGate', 'e41_bbbc047_hypergate_augbest40_seed0'),
        10: ('MVC_HyperGate', 'e41_bbbc047_hypergate_augbest40_seed10'),
        100: ('MVC_HyperGate', 'e41_bbbc047_hypergate_augbest40_seed100'),
        1000: ('MVC_HyperGate', 'e41_bbbc047_hypergate_augbest40_seed1000'),
    },
    'MVCPert': {
        3407: E40_MODAL_ABLATION_ROOT / 'test_restruction_result_all_samples.csv',
        0: ('MVC_HyperGateResidualVAE', 'e42_transmorph_corrected_main_seed0'),
        10: ('MVC_HyperGateResidualVAE', 'e42_transmorph_corrected_main_seed10'),
        100: ('MVC_HyperGateResidualVAE', 'e42_transmorph_corrected_main_seed100'),
        1000: ('MVC_HyperGateResidualVAE', 'e42_transmorph_corrected_main_seed1000'),
    },
}
GE_METRICS = ['systema-GE PCC', 'GE PCC', 'DEG-GE PCC']
CP_METRICS = ['systema-CP PCC', 'CP PCC', 'DEG-CP PCC']
RMSE_METRICS = ['GE RMSE', 'CP RMSE']
RMSE_LINE_STYLE = {
    'GE RMSE': {'color': FIG2_PALETTE['green'], 'lw': 2.0, 'ls': '--', 'marker': 'o', 'label': 'GE RMSE'},
    'CP RMSE': {'color': FIG2_PALETTE['terracotta'], 'lw': 2.0, 'ls': '-', 'marker': 's', 'label': 'CP RMSE'},
}
T_CRITICAL_95_BY_DF = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183661,
    6: 2.44691185114,
    7: 2.36462425101,
    8: 2.3060041352,
    9: 2.26215716285,
    10: 2.22813885196,
}
MANUSCRIPT_FIGURE_DIR = os.environ.get("MVCPERT_MANUSCRIPT_FIGURE_DIR")

# ---- Line styles for trajectory panel (Panel A) ----
RECON_METRIC_COLORS = {
    'GE system': FIG2_PALETTE['green'],
    'GE profile': '#78AF7F',
    'GE-DEG': '#A9C7B0',
    'CP system': FIG2_PALETTE['terracotta'],
    'CP profile': '#E08A5C',
    'CP-DEG': '#F2C59E',
}
METRIC_LINE_STYLE = {
    'systema-GE PCC': {'color': RECON_METRIC_COLORS['GE system'], 'lw': 2.2, 'ls': '--', 'marker': 'o', 'zorder': 5, 'label': 'GE system'},
    'GE PCC':         {'color': RECON_METRIC_COLORS['GE profile'], 'lw': 1.6, 'ls': '--', 'marker': 's', 'zorder': 4, 'label': 'GE profile'},
    'DEG-GE PCC':     {'color': RECON_METRIC_COLORS['GE-DEG'], 'lw': 1.3, 'ls': '--', 'marker': '^', 'zorder': 3, 'label': 'GE-DEG'},
    'systema-CP PCC': {'color': RECON_METRIC_COLORS['CP system'], 'lw': 2.2, 'ls': '-',  'marker': 'o', 'zorder': 5, 'label': 'CP system'},
    'CP PCC':         {'color': RECON_METRIC_COLORS['CP profile'], 'lw': 1.6, 'ls': '-',  'marker': 's', 'zorder': 4, 'label': 'CP profile'},
    'DEG-CP PCC':     {'color': RECON_METRIC_COLORS['CP-DEG'], 'lw': 1.3, 'ls': '-',  'marker': '^', 'zorder': 3, 'label': 'CP-DEG'},
}
PANEL_E_METRIC_COLORS = {
    **RECON_METRIC_COLORS,
}
PANEL_F_METRIC_COLORS = {
    'Informative overlap@20': FIG2_PALETTE['blue'],
    'Informative overlap@50': '#7BA4D0',
    'Informative overlap@100': '#AED0EA',
    'Top20 sign recall': FIG2_PALETTE['violet'],
    'Top50 sign recall': '#A4A3CB',
    'Top100 sign recall': FIG2_PALETTE['lilac'],
}

# Panel A: concise 3-state architecture trajectory
TRAJECTORY_VARIANT_ORDER = ['Baseline', 'Gate', 'HyperGate', 'MVCPert']
TRAJECTORY_VARIANT_LABELS = {
    'Baseline': 'MVCPert-Base',
    'Gate': '+ Gating',
    'HyperGate': '+ Hypergraph',
    'MVCPert': '+ Refiner',
}
TRAJECTORY_STEPS = [
    ('+Gate', 'Baseline', 'Gate'),
    ('+Hypergraph', 'Gate', 'HyperGate'),
    ('+Residual VAE', 'HyperGate', 'MVCPert'),
]

# ---- Heatmap colours (Panel B) ----
HEATMAP_CMAP = 'RdBu_r'
DISPLAY_SCALE = 1e3
DISPLAY_SCALE_LABEL = r'($\times 10^{-3}$)'

AXIS_LABEL_FS = 13
TITLE_FS = 13
TICK_LABEL_FS = 10.5
LEGEND_FS = 10.5
GROUP_LABEL_FS = 10.5
PANEL_LABEL_FS = 16
SMALL_TEXT_FS = 11


def add_panel_label_large(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=PANEL_LABEL_FS,
        fontweight='bold',
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ('pdf',):
        path = OUTPUT_DIR / f'{stem}.{suffix}'
        fig.savefig(path)
        written.append(path)
    if MANUSCRIPT_FIGURE_DIR:
        manuscript_figure_dir = Path(MANUSCRIPT_FIGURE_DIR)
        manuscript_figure_dir.mkdir(parents=True, exist_ok=True)
        mirrored: list[Path] = []
        for path in written:
            mirror_path = manuscript_figure_dir / path.name
            shutil.copy2(path, mirror_path)
            mirrored.append(mirror_path)
        return written + mirrored
    return written


def init_plot_env() -> None:
    _shared_init_plot_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def _format_display_number(value: float, decimals: int = 1, signed: bool = False) -> str:
    text = f'{value:+.{decimals}f}' if signed else f'{value:.{decimals}f}'
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text


def _tick_formatter(decimals: int = 1) -> mticker.FuncFormatter:
    return mticker.FuncFormatter(lambda value, _: _format_display_number(value, decimals=decimals, signed=False))


def _text_color_for_fill(fill_color: str) -> str:
    fill = fill_color.lstrip('#')
    if len(fill) != 6:
        return '#222222'
    r = int(fill[0:2], 16) / 255.0
    g = int(fill[2:4], 16) / 255.0
    b = int(fill[4:6], 16) / 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if luminance < 0.62 else '#222222'


def _normalize_display_metric_name(metric: str) -> str:
    mapping = {
        'GE systema': 'GE system',
        'CP systema': 'CP system',
        'DEG-GE PCC': 'GE-DEG',
        'DEG-CP PCC': 'CP-DEG',
    }
    return mapping.get(metric, metric)


def _mean_profile_from_df(df: pd.DataFrame) -> dict[str, float]:
    def mean_or_nan(column: str) -> float:
        return float(df[column].mean()) if column in df.columns else np.nan

    return {
        'systema-GE PCC': mean_or_nan('systema_pearson_GE'),
        'GE PCC': mean_or_nan('ge_pred_pearson'),
        'DEG-GE PCC': mean_or_nan('DEG_ge_pred_pearson'),
        'GE RMSE': mean_or_nan('ge_pred_rmse'),
        'systema-CP PCC': mean_or_nan('systema_pearson_CP'),
        'CP PCC': mean_or_nan('cp_pred_pearson'),
        'DEG-CP PCC': mean_or_nan('DEG_cp_pred_pearson'),
        'CP RMSE': mean_or_nan('cp_pred_rmse'),
    }


def _latest_multiseed_predict_path(model_type: str, output_tag: str) -> Path:
    roots = [
        STRUCTURE_MULTISEED_ROOT,
        ARTIFACT_ROOT / 'BBBC047/followups/E42_bbbc047_transmorph_variance_controls/formal_train',
        ARTIFACT_ROOT / 'BBBC047/followups/E43_bbbc047_gatedpcc_multiseed_ci/formal_train',
    ]
    pattern = f'*/train_outputs/*/{model_type}_{output_tag}_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv'
    matches = []
    for root in roots:
        if root.exists():
            matches.extend(root.glob(pattern))
    matches = sorted(matches)
    if not matches:
        raise FileNotFoundError(f'no multiseed match for {model_type} / {output_tag}')
    return matches[-1]


def load_structure_ci_seed_table() -> pd.DataFrame:
    rows = []
    for variant, seed_specs in STRUCTURE_CI_TAGS.items():
        for seed in SEED_ORDER:
            spec = seed_specs[seed]
            path = _latest_multiseed_predict_path(*spec) if isinstance(spec, tuple) else spec
            if not path.exists():
                raise FileNotFoundError(f'missing seed result for {variant} seed {seed}: {path}')
            metrics = _mean_profile_from_df(pd.read_csv(path))
            rows.append({'Variant': variant, 'Seed': seed, **metrics})
    return pd.DataFrame(rows)


def summarize_structure_ci(seed_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in TRAJECTORY_VARIANT_ORDER:
        subset = seed_table[seed_table['Variant'] == variant]
        row = {'Variant': variant, 'n_seeds': int(len(subset))}
        for metric in STRUCTURE_PCC_METRICS + RMSE_METRICS:
            values = subset[metric].astype(float).to_numpy()
            row[f'{metric} mean'] = float(values.mean())
            row[f'{metric} std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_ci_bounds(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(arr.size)
    critical = T_CRITICAL_95_BY_DF.get(arr.size - 1, 1.96)
    margin = critical * sem
    return mean, mean - margin, mean + margin


def build_structure_delta_long(seed_table: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    baseline = seed_table[seed_table['Variant'] == 'Baseline'].set_index('Seed')
    for variant in ['Gate', 'HyperGate', 'MVCPert']:
        sub = seed_table[seed_table['Variant'] == variant].set_index('Seed')
        for seed in SEED_ORDER:
            for metric in metrics:
                rows.append(
                    {
                        'Variant': variant,
                        'Seed': seed,
                        'Metric': metric,
                        'Delta': float(sub.loc[seed, metric] - baseline.loc[seed, metric]),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading for dumbbell charts
# ---------------------------------------------------------------------------

def _load_recon_dumbbell() -> pd.DataFrame:
    """Return rows: Model, Metric, NoAug, WithAug for key reconstruction metrics."""
    profile_specs = {
        'Baseline': {
            'NoAug': ARTIFACT_ROOT / '2026-04-08/MVC_splitlock_official_v1_20260408_baseline_noaug40_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
            'WithAug': ARTIFACT_ROOT / '2026-04-08/MVC_splitlock_official_v1_20260408_baseline_augbest40_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        },
        'HyperGate': {
            'NoAug': ARTIFACT_ROOT / '2026-04-08/MVC_Dynamics_Modulation_splitlock_official_v1_20260408_dynamics_noaug40_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
            'WithAug': ARTIFACT_ROOT / '2026-04-08/MVC_Dynamics_Modulation_splitlock_official_v1_20260408_dynamics_augbest40_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        },
        'MVCPert': {
            'NoAug': ARTIFACT_ROOT / 'BBBC047/followups/E42_bbbc047_transmorph_variance_controls/formal_train/20260430_141436/train_outputs/2026-04-30/MVC_HyperGateResidualVAE_e42_transmorph_no_augmentation_seed3407_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
            'WithAug': ARTIFACT_ROOT / 'BBBC047/followups/E42_bbbc047_transmorph_variance_controls/formal_train/20260430_141436/train_outputs/2026-04-30/MVC_HyperGateResidualVAE_e42_transmorph_corrected_main_seed3407_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        },
    }
    metric_cols = {
        'GE system': 'systema_pearson_GE',
        'GE profile': 'ge_pred_pearson',
        'GE-DEG': 'DEG_ge_pred_pearson',
        'CP system': 'systema_pearson_CP',
        'CP profile': 'cp_pred_pearson',
        'CP-DEG': 'DEG_cp_pred_pearson',
    }

    rows = []
    for model_label, spec in profile_specs.items():
        noaug_df = pd.read_csv(spec['NoAug'])
        aug_df = pd.read_csv(spec['WithAug'])
        for metric, col in metric_cols.items():
            rows.append({
                'Model': model_label,
                'Metric': metric,
                'NoAug': float(noaug_df[col].mean()),
                'WithAug': float(aug_df[col].mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel plotting helpers
# ---------------------------------------------------------------------------

def plot_multimodal_gain_panel(ax: plt.Axes, modal_gain: pd.DataFrame) -> None:
    gain_rows = modal_gain.copy()
    gain_rows['Metric'] = gain_rows['Metric'].map(_normalize_display_metric_name)
    gain_rows['PlotColor'] = gain_rows['Metric'].map(PANEL_E_METRIC_COLORS).fillna(gain_rows['Color'])
    y_positions = {
        'GE system': 6.0,
        'GE profile': 5.0,
        'GE-DEG': 4.0,
        'CP system': 2.0,
        'CP profile': 1.0,
        'CP-DEG': 0.0,
    }
    y = gain_rows['Metric'].map(y_positions).to_numpy(dtype=float)
    short_labels = {
        'GE system': 'GE system',
        'GE profile': 'GE profile',
        'GE-DEG': 'GE-DEG',
        'CP system': 'CP system',
        'CP profile': 'CP profile',
        'CP-DEG': 'CP-DEG',
    }
    ax.barh(
        y,
        gain_rows['Delta_x1e3'],
        color=gain_rows['PlotColor'],
        height=0.62,
        alpha=0.90,
        edgecolor='white',
        linewidth=0.8,
    )
    ax.axhline(3.0, color=FIG2_NEUTRAL_LIGHT, lw=0.9, ls='--')
    ax.axvline(0.0, color=FIG2_NEUTRAL_LINE, lw=0.9, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([short_labels[m] for m in gain_rows['Metric']])
    ax.set_xlabel('Improvement over unimodal training (ΔPCC ×10⁻³)', fontsize=AXIS_LABEL_FS)
    xmin = min(-1, float(gain_rows['Delta_x1e3'].min()) - 0.20)
    xmax = max(8.8, float(gain_rows['Delta_x1e3'].max()) + 0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.75, 6.75)
    ax.set_title('Multimodal gain in reconstruction', fontsize=TITLE_FS, fontweight='bold', pad=10)
    ax.text(
        0.11,
        6.42,
        'GE-related',
        transform=ax.get_yaxis_transform(),
        ha='left',
        va='bottom',
        fontsize=GROUP_LABEL_FS,
        fontweight='bold',
        color='#333333',
    )
    ax.text(
        0.11,
        2.42,
        'CP-related',
        transform=ax.get_yaxis_transform(),
        ha='left',
        va='bottom',
        fontsize=GROUP_LABEL_FS,
        fontweight='bold',
        color='#333333',
    )
    add_panel_label_large(ax, 'E')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FS)
    clean_axes(ax)


def plot_ge_only_gain_panel(ax: plt.Axes, ge_only_downstream_gain: pd.DataFrame) -> None:
    downstream_rows = ge_only_downstream_gain.copy()
    downstream_rows['PlotColor'] = downstream_rows['Metric'].map(PANEL_F_METRIC_COLORS).fillna(downstream_rows['Color'])
    y_positions = {
        'Informative overlap@20': 6.0,
        'Informative overlap@50': 5.0,
        'Informative overlap@100': 4.0,
        'Top20 sign recall': 2.0,
        'Top50 sign recall': 1.0,
        'Top100 sign recall': 0.0,
    }
    y = downstream_rows['Metric'].map(y_positions).to_numpy(dtype=float)
    short_labels = {
        'Informative overlap@20': '@20',
        'Informative overlap@50': '@50',
        'Informative overlap@100': '@100',
        'Top20 sign recall': '@20',
        'Top50 sign recall': '@50',
        'Top100 sign recall': '@100',
    }
    ax.barh(
        y,
        downstream_rows['Delta_pct'],
        color=downstream_rows['PlotColor'],
        height=0.60,
        alpha=0.90,
        edgecolor='white',
        linewidth=0.8,
    )
    ax.axhline(3.0, color=FIG2_NEUTRAL_LIGHT, lw=0.9, ls='--')
    ax.axvline(0.0, color=FIG2_NEUTRAL_LINE, lw=0.9, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([short_labels[m] for m in downstream_rows['Metric']])
    ax.set_xlabel('Improvement over unimodal training (%)', fontsize=AXIS_LABEL_FS)
    ax.set_xlim(0.0, max(7.2, float(downstream_rows['Delta_pct'].max()) + 0.8))
    ax.set_ylim(-0.75, 6.75)
    ax.set_title('Multimodal gain in response recovery', fontsize=TITLE_FS, fontweight='bold', pad=10)
    ax.text(
        0.02,
        6.42,
        'Overlap',
        transform=ax.get_yaxis_transform(),
        ha='left',
        va='bottom',
        fontsize=GROUP_LABEL_FS,
        fontweight='bold',
        color='#333333',
    )
    ax.text(
        0.02,
        2.42,
        'Sign recall',
        transform=ax.get_yaxis_transform(),
        ha='left',
        va='bottom',
        fontsize=GROUP_LABEL_FS,
        fontweight='bold',
        color='#333333',
    )
    add_panel_label_large(ax, 'F')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FS)
    clean_axes(ax)

def plot_metric_family(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metrics: list[str],
    panel_label: str,
    title: str,
    ylabel: str | None = None,
) -> None:
    """Plot delta-from-baseline mean ± std trajectories for one metric family."""
    df = summary.set_index('Variant').loc[TRAJECTORY_VARIANT_ORDER]
    x = np.arange(len(TRAJECTORY_VARIANT_ORDER))
    labels = [TRAJECTORY_VARIANT_LABELS[v] for v in TRAJECTORY_VARIANT_ORDER]
    baseline_row = df.loc['Baseline']

    for metric in metrics:
        style = METRIC_LINE_STYLE[metric]
        means = df[f'{metric} mean'].values - baseline_row[f'{metric} mean']
        stds = df[f'{metric} std'].values
        ax.fill_between(
            x,
            means - stds,
            means + stds,
            color=style['color'],
            alpha=0.12,
            linewidth=0,
            zorder=style['zorder'] - 1,
        )
        ax.plot(
            x,
            means,
            color=style['color'],
            lw=style['lw'],
            ls=style['ls'],
            marker=style['marker'],
            markersize=8.0,
            markeredgecolor='white',
            markeredgewidth=0.4,
            label=style['label'],
            zorder=style['zorder'],
        )

    ax.axhline(0, color=FIG2_NEUTRAL_LINE, lw=0.7, alpha=0.7, zorder=1)
    y_min = min((df[f'{m} mean'] - baseline_row[f'{m} mean'] - df[f'{m} std']).min() for m in metrics)
    y_max = max((df[f'{m} mean'] - baseline_row[f'{m} mean'] + df[f'{m} std']).max() for m in metrics)
    pad = max(0.0015, (y_max - y_min) * 0.12)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', pad=8)
    add_panel_label_large(ax, panel_label)
    clean_axes(ax)
    legend_handles = [
        plt.Line2D([0], [0], color=METRIC_LINE_STYLE[metric]['color'],
                   lw=METRIC_LINE_STYLE[metric]['lw'],
                   ls=METRIC_LINE_STYLE[metric]['ls'],
                   marker=METRIC_LINE_STYLE[metric]['marker'],
                   markersize=7.0,
                   markeredgecolor='white', markeredgewidth=0.35,
                   label=METRIC_LINE_STYLE[metric]['label'])
        for metric in metrics
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper left',
        ncol=1,
        fontsize=LEGEND_FS,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=1.0,
    )


def plot_metric_family_boxplot(
    ax: plt.Axes,
    delta_df: pd.DataFrame,
    metrics: list[str],
    panel_label: str,
    title: str,
    ylabel: str | None = None,
    legend_ax: plt.Axes | None = None,
) -> None:
    """Grouped boxplots for delta-from-baseline seed distributions."""
    delta_df = delta_df.copy()
    delta_df['Delta'] = delta_df['Delta'] * DISPLAY_SCALE

    variants = ['Gate', 'HyperGate', 'MVCPert']
    box_step = 0.041
    width = 0.038
    group_starts = {'Gate': 0.0, 'HyperGate': 0.175, 'MVCPert': 0.350}
    positions_by_variant = {
        variant: group_starts[variant] + np.array([0.0, box_step, 2 * box_step])
        for variant in variants
    }
    tick_positions = [positions_by_variant[variant].mean() for variant in variants]

    ymin = float(delta_df['Delta'].min())
    ymax = float(delta_df['Delta'].max())
    ax.axhline(0, color=FIG2_NEUTRAL_LINE, lw=1.0, ls='--', alpha=0.85, zorder=1)

    for metric_idx, metric in enumerate(metrics):
        style = METRIC_LINE_STYLE[metric]
        data = [
            delta_df[(delta_df['Variant'] == variant) & (delta_df['Metric'] == metric)]['Delta'].to_list()
            for variant in variants
        ]
        positions = [positions_by_variant[variant][metric_idx] for variant in variants]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={'color': '#3E3E3E', 'linewidth': 1.15},
            whiskerprops={'color': '#555555', 'linewidth': 0.95},
            capprops={'color': '#555555', 'linewidth': 0.95},
            boxprops={'edgecolor': '#555555', 'linewidth': 0.95},
            zorder=3,
        )
        for patch in bp['boxes']:
            patch.set_facecolor(style['color'])
            patch.set_alpha(0.62)

    y_range = ymax - ymin
    lower_pad = max(0.00035, y_range * 0.05)
    upper_pad = max(0.00055, y_range * 0.07)
    zero_pad = max(0.0012, y_range * 0.06)
    y_lower = min(ymin - lower_pad, -zero_pad)
    ax.set_ylim(y_lower, ymax + upper_pad)
    ax.set_xlim(-0.020, positions_by_variant['MVCPert'][-1] + 0.020)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([TRAJECTORY_VARIANT_LABELS[v] for v in variants], fontsize=TICK_LABEL_FS)
    ax.set_xlabel('Added components over MVCPert-Base', fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis='x', pad=1.4)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FS)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='lower'))
    ax.yaxis.set_major_formatter(_tick_formatter(decimals=1))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', pad=8)
    add_panel_label_large(ax, panel_label)
    if legend_ax is not None:
        draw_boxed_metric_legend(legend_ax, metrics)
    else:
        ax.legend(
            handles=[
                Patch(facecolor=METRIC_LINE_STYLE[m]['color'], edgecolor='#555555', linewidth=0.9, label=METRIC_LINE_STYLE[m]['label'])
                for m in metrics
            ],
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            fontsize=LEGEND_FS,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor='#8A8A8A',
            facecolor='white',
        )
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)


def draw_boxed_metric_legend(ax: plt.Axes, metrics: list[str]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    frame = Rectangle(
        (0.00, 0.295),
        0.98,
        0.41,
        facecolor='white',
        edgecolor='#8A8A8A',
        linewidth=0.9,
        joinstyle='miter',
    )
    ax.add_patch(frame)

    if len(metrics) == 3:
        y_positions = [0.61, 0.50, 0.39]
    else:
        y_positions = [0.74, 0.62, 0.50, 0.34, 0.22, 0.10]
    for y, metric in zip(y_positions, metrics):
        style = METRIC_LINE_STYLE[metric]
        ax.add_patch(
            Rectangle(
                (0.10, y - 0.022),
                0.085,
                0.044,
                facecolor=style['color'],
                edgecolor='#555555',
                linewidth=0.85,
            )
        )
        ax.text(
            0.23,
            y,
            style['label'],
            va='center',
            ha='left',
            fontsize=LEGEND_FS,
        )


def plot_dumbbell(
    ax: plt.Axes,
    df: pd.DataFrame,
    metrics_order: list[str],
    title: str,
    xlim: tuple[float, float],
    panel_label: str | None = None,
) -> None:
    """Grouped augmentation-delta dot plot."""
    centers = np.arange(len(metrics_order))[::-1]
    offsets = {'Baseline': -0.26, 'HyperGate': 0.00, 'MVCPert': 0.26}
    xlim = (xlim[0] * DISPLAY_SCALE, xlim[1] * DISPLAY_SCALE)

    # Compute delta from absolute values
    df = df.copy()
    df['Metric'] = df['Metric'].map(_normalize_display_metric_name)
    df['Delta'] = (df['WithAug'] - df['NoAug']) * DISPLAY_SCALE

    for model in AUGMENT_MODEL_ORDER:
        color = FIG2_MODEL_COLORS.get(model, METHOD_COLORS.get(model, '#333333'))
        sub = df[df['Model'] == model].set_index('Metric').loc[metrics_order]
        y = centers + offsets[model]
        deltas = sub['Delta'].values

        # Keep only faint reference lines so the deltas read as grouped dots.
        ax.hlines(y, 0, deltas, colors=color, linewidth=1.2, alpha=0.32, zorder=2)
        ax.scatter(
            deltas,
            y,
            s=82,
            c=color,
            edgecolors='white',
            linewidths=0.65,
            zorder=4,
            label=PANEL_D_LEGEND_LABELS.get(model, model),
        )

    # Zero line
    ax.axvline(0, color=FIG2_NEUTRAL_LINE, lw=0.7, ls='-', alpha=0.6, zorder=1)

    ax.set_yticks(centers)
    ax.set_yticklabels(metrics_order, fontsize=TICK_LABEL_FS)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', pad=8)
    ax.set_xlabel(f'ΔPCC with augmentation (×10⁻³)', fontsize=AXIS_LABEL_FS)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(_tick_formatter(decimals=1))
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    if panel_label is not None:
        add_panel_label_large(ax, panel_label)
    clean_axes(ax)
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=9.0,
               markerfacecolor=FIG2_MODEL_COLORS.get(model, METHOD_COLORS.get(model, '#333333')),
               markeredgecolor='white', markeredgewidth=0.6, label=PANEL_D_LEGEND_LABELS.get(model, model))
        for model in AUGMENT_MODEL_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc='lower right',
        bbox_to_anchor=(0.985, 0.03),
        ncol=1,
        fontsize=LEGEND_FS,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='#8A8A8A',
        facecolor='white',
        handlelength=0.8,
        handletextpad=0.45,
        borderpad=0.45,
    )


def plot_rmse_vs_baseline(
    ax: plt.Axes,
    structure_seed_table: pd.DataFrame,
    title: str,
) -> None:
    variant_order = TRAJECTORY_VARIANT_ORDER
    x = np.arange(len(variant_order))
    labels = [TRAJECTORY_VARIANT_LABELS[variant] for variant in variant_order]
    baseline = structure_seed_table[structure_seed_table['Variant'] == 'Baseline'].set_index('Seed').loc[SEED_ORDER]

    y_lows: list[float] = []
    y_highs: list[float] = []
    for metric in RMSE_METRICS:
        style = RMSE_LINE_STYLE[metric]
        means = []
        ci_lows = []
        ci_highs = []
        for variant in variant_order:
            if variant == 'Baseline':
                reductions = np.zeros(len(SEED_ORDER), dtype=np.float64)
            else:
                variant_values = (
                    structure_seed_table[structure_seed_table['Variant'] == variant]
                    .set_index('Seed')
                    .loc[SEED_ORDER, metric]
                    .to_numpy(dtype=np.float64)
                )
                baseline_values = baseline[metric].to_numpy(dtype=np.float64)
                reductions = baseline_values - variant_values
            mean, ci_low, ci_high = _mean_ci_bounds(reductions)
            means.append(mean * DISPLAY_SCALE)
            ci_lows.append(ci_low * DISPLAY_SCALE)
            ci_highs.append(ci_high * DISPLAY_SCALE)

        means_arr = np.asarray(means, dtype=np.float64)
        ci_lows_arr = np.asarray(ci_lows, dtype=np.float64)
        ci_highs_arr = np.asarray(ci_highs, dtype=np.float64)
        y_lows.extend(ci_lows_arr.tolist())
        y_highs.extend(ci_highs_arr.tolist())
        ax.fill_between(
            x,
            ci_lows_arr,
            ci_highs_arr,
            color=style['color'],
            alpha=0.16,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            x,
            means_arr,
            color=style['color'],
            lw=style['lw'],
            ls=style['ls'],
            marker=style['marker'],
            markersize=6.2,
            markeredgecolor='white',
            markeredgewidth=0.45,
            label=style['label'],
            zorder=3,
        )

    y_min = float(np.nanmin(y_lows))
    y_max = float(np.nanmax(y_highs))
    pad = max(0.8, (y_max - y_min) * 0.12)
    ax.axhline(0, color=FIG2_NEUTRAL_LINE, lw=0.8, alpha=0.75, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FS)
    ax.set_ylim(min(-0.8, y_min - pad), y_max + pad)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold')
    ax.set_ylabel(f'RMSE reduction vs. MVCPert-Base {DISPLAY_SCALE_LABEL}', fontsize=AXIS_LABEL_FS)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(_tick_formatter(decimals=1))
    add_panel_label_large(ax, 'C')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    clean_axes(ax)
    ax.legend(
        loc='upper left',
        fontsize=LEGEND_FS,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='#8A8A8A',
        facecolor='white',
        borderpad=0.45,
        handlelength=1.0,
        handletextpad=0.4,
        bbox_to_anchor=(0.02, 0.98),
    )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def _compute_step_deltas(structure: pd.DataFrame) -> pd.DataFrame:
    """Step deltas for the 3-step trajectory (no HG-no-graph)."""
    keyed = structure.set_index('Variant')
    rows = []
    for order, (step, start, end) in enumerate(TRAJECTORY_STEPS):
        row = {'Step': step, 'Start': start, 'End': end, 'Order': order}
        for metric in STRUCTURE_PCC_METRICS + ['GE RMSE', 'CP RMSE']:
            row[metric] = float(keyed.loc[end, metric] - keyed.loc[start, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def _load_local_analysis_csv(name: str) -> pd.DataFrame | None:
    path = LOCAL_ANALYSIS_DIR / name
    if path.exists():
        return pd.read_csv(path).replace(LEGACY_LABEL_MAP)
    return None


def plot_figure2() -> None:
    structure_seed_table = _load_local_analysis_csv('figure3_structure_multiseed_seed_table.csv')
    structure_ge_delta = _load_local_analysis_csv('figure3_structure_ge_delta_boxplot_table.csv')
    structure_cp_delta = _load_local_analysis_csv('figure3_structure_cp_delta_boxplot_table.csv')
    recon_dumb = _load_local_analysis_csv('figure3_augmentation_reconstruction_absolute.csv')
    modal_gain = _load_local_analysis_csv('figure3_modal_gain_error_analysis.csv')
    ge_only_downstream_gain = _load_local_analysis_csv('figure3_ge_only_downstream_gain.csv')

    if structure_seed_table is None:
        raise FileNotFoundError(f'missing local analysis table: {LOCAL_ANALYSIS_DIR / "figure3_structure_multiseed_seed_table.csv"}')
    if structure_ge_delta is None:
        raise FileNotFoundError(f'missing local analysis table: {LOCAL_ANALYSIS_DIR / "figure3_structure_ge_delta_boxplot_table.csv"}')
    if structure_cp_delta is None:
        raise FileNotFoundError(f'missing local analysis table: {LOCAL_ANALYSIS_DIR / "figure3_structure_cp_delta_boxplot_table.csv"}')
    if recon_dumb is None:
        raise FileNotFoundError(f'missing local analysis table: {LOCAL_ANALYSIS_DIR / "figure3_augmentation_reconstruction_absolute.csv"}')
    if modal_gain is None:
        raise FileNotFoundError(f'missing local analysis table: {LOCAL_ANALYSIS_DIR / "figure3_modal_gain_error_analysis.csv"}')
    if ge_only_downstream_gain is None:
        raise FileNotFoundError(f'missing local analysis table: {LOCAL_ANALYSIS_DIR / "figure3_ge_only_downstream_gain.csv"}')

    fig = plt.figure(figsize=(12.8, 11.2))
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.92, 0.92, 0.92],
        wspace=0.20, hspace=0.44,
    )
    top_left = gs[0, 0].subgridspec(1, 2, width_ratios=[0.6, 0.18], wspace=0.12)
    top_right = gs[0, 1].subgridspec(1, 2, width_ratios=[0.6, 0.18], wspace=0.12)

    # ---- Panel A: GE-side seed-level delta from MVCPert-Base ----
    ax = fig.add_subplot(top_left[0, 0])
    ax_leg = fig.add_subplot(top_left[0, 1])
    plot_metric_family_boxplot(
        ax,
        structure_ge_delta,
        GE_METRICS,
        panel_label='A',
        title='GE response recovery',
        ylabel=f'ΔPCC vs. MVCPert-Base {DISPLAY_SCALE_LABEL}',
        legend_ax=ax_leg,
    )

    # ---- Panel B: CP-side seed-level delta from MVCPert-Base ----
    ax2 = fig.add_subplot(top_right[0, 0])
    ax2_leg = fig.add_subplot(top_right[0, 1])
    plot_metric_family_boxplot(
        ax2,
        structure_cp_delta,
        CP_METRICS,
        panel_label='B',
        title='CP response recovery',
        legend_ax=ax2_leg,
    )

    # ---- Panel C: RMSE vs MVCPert-Base ----
    ax3 = fig.add_subplot(gs[1, 0])
    plot_rmse_vs_baseline(
        ax3,
        structure_seed_table,
        title='Error reduction trajectory',
    )

    ax3.set_xticklabels(['MVCPert-Base', '+ Gating', '+ Hypergraph', '+ Refiner'], fontsize=TICK_LABEL_FS)

    # ---- Panel D: Augmentation delta plot ----
    ax4 = fig.add_subplot(gs[1, 1])
    plot_dumbbell(
        ax4, recon_dumb,
        metrics_order=['GE system', 'GE profile', 'GE-DEG', 'CP system', 'CP profile', 'CP-DEG'],
        title='Augmentation effect on reconstruction',
        xlim=(-0.004, 0.018),
        panel_label='D',
    )

    # ---- Panel E: multimodal gain over unimodal ----
    ax5 = fig.add_subplot(gs[2, 0])
    plot_multimodal_gain_panel(ax5, modal_gain)

    # ---- Panel F: multimodal gain over GE-only response recovery ----
    ax6 = fig.add_subplot(gs[2, 1])
    plot_ge_only_gain_panel(ax6, ge_only_downstream_gain)

    fig.subplots_adjust(top=0.95, left=0.06, right=0.985, bottom=0.06)
    save_figure(fig, 'figure3_bbbc047_architecture_ablation')
    plt.close(fig)


def main() -> None:
    init_plot_env()
    plot_figure2()


if __name__ == '__main__':
    main()
