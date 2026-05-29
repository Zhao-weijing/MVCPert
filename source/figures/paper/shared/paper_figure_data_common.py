#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from paper_plot_style import ANALYSIS_DIR, FIG_DIR, PALETTE, apply_publication_style

RELEASE_SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("MVCPERT_ARTIFACT_ROOT", "<ARTIFACT_ROOT>"))
AIDD = Path(os.environ.get("MVCPERT_AIDD_ROOT", str(RELEASE_SOURCE_ROOT)))
E40_MODAL_ABLATION_ROOT = ROOT / 'BBBC047/followups/E40_hypergate_residual_vae_modal_ablation_v1/formal_train/20260427_1258/train_outputs/2026-04-27/MVC_HyperGateResidualVAE_e40_factorized_roles_infonce001_remainder005_modal_ablation_v1_BBBC047_smiles_split/ECFP4_Default/predict'
E40_CLEAN_UNIMODAL_ROOT = ROOT / 'BBBC047/followups/E40_hypergate_residual_vae_clean_unimodal_v1/formal_train/20260427_213011/train_outputs/2026-04-27'

BBBC047_PROFILES = {
    'Baseline': ROOT / 'BBBC047/followups/E26_bbbc047_baseline_to_hypergate_7run_ablation/formal_train/20260421_debug/train_outputs/2026-04-21/MVC_e26_bbbc047_baseline_to_hypergate_7run_baseline_augbest40_v1_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
    'MVCPert-HyperGate': ROOT / 'BBBC047/followups/E26_bbbc047_baseline_to_hypergate_7run_ablation/formal_train/20260421_debug/train_outputs/2026-04-21/MVC_HyperGate_e26_bbbc047_baseline_to_hypergate_7run_hypergate_augbest40_v1_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
    'MVCPert': E40_MODAL_ABLATION_ROOT / 'test_restruction_result_all_samples.csv',
}

BBBC047_CLEAN_UNIMODAL_PROFILES = {
    'Dual MVCPert': E40_MODAL_ABLATION_ROOT / 'test_restruction_result_all_samples.csv',
    'GE-only target': E40_CLEAN_UNIMODAL_ROOT / 'MVC_HyperGateResidualVAE_NoCP_e40_clean_ge_only_target_v1_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
    'CP-only target': E40_CLEAN_UNIMODAL_ROOT / 'MVC_HyperGateResidualVAE_NoGE_e40_clean_cp_only_target_v1_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
}

BBBC047_STRUCTURE_PROFILES = {
    'Baseline': BBBC047_PROFILES['Baseline'],
    'Gate': ROOT / 'BBBC047/followups/E26_bbbc047_baseline_to_hypergate_7run_ablation/formal_train/20260421_debug/train_outputs/2026-04-21/MVC_GatedPCC_e26_bbbc047_baseline_to_hypergate_7run_gatedpcc_augbest40_v1_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
    'HyperGate (no graph)': ROOT / 'BBBC047/followups/E27_bbbc047_hypergate_no_hyper_refinement_v1/formal_train/20260421_112615/train_outputs/2026-04-21/MVC_HyperGate_e27_bbbc047_hypergate_nohyper_augbest40_v1_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
    'HyperGate': BBBC047_PROFILES['MVCPert-HyperGate'],
    'MVCPert': E40_MODAL_ABLATION_ROOT / 'test_restruction_result_all_samples.csv',
}

STRUCTURE_VARIANT_ORDER = ['Baseline', 'Gate', 'HyperGate (no graph)', 'HyperGate', 'MVCPert']
STRUCTURE_VARIANT_LABELS = {
    'Baseline': 'Baseline',
    'Gate': '+Gate',
    'HyperGate (no graph)': 'HyperGate\n(no graph)',
    'HyperGate': 'HyperGate',
    'MVCPert': 'MVCPert',
}
STRUCTURE_STEP_DEFINITIONS = [
    ('Gate', 'Baseline', 'Gate'),
    ('HyperGate core', 'Gate', 'HyperGate (no graph)'),
    ('Hypergraph', 'HyperGate (no graph)', 'HyperGate'),
    ('Residual VAE', 'HyperGate', 'MVCPert'),
]
STRUCTURE_PCC_METRICS = ['systema-GE PCC', 'GE PCC', 'DEG-GE PCC', 'systema-CP PCC', 'CP PCC', 'DEG-CP PCC']

AUGMENTATION_METRIC_DELTA = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-09_full_official_v1_augbest_compare/aug_vs_noaug_metric_delta.csv'
AUGMENTATION_DOWNSTREAM_DELTA = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-09_full_official_v1_augbest_compare/aug_vs_noaug_downstream_delta.csv'

EXTERNAL_PROFILES = {
    'BBBC036': {
        'Baseline': ROOT / 'BBBC036/followups/E21_bbbc036_followups/weakaug_train/20260418_115747/train_outputs/2026-04-18/MVC_e21_baseline_main_BBBC036_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        'MVCPert-HyperGate': ROOT / 'BBBC036/followups/E21_bbbc036_followups/weakaug_train/20260418_115747/train_outputs/2026-04-18/MVC_HyperGate_e21_hypergate_main_BBBC036_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        'MVCPert': ROOT / 'BBBC036/followups/E40_bbbc036_same_moa_eval/20260424_1/train_outputs/2026-04-24/MVC_HyperGateResidualVAE_e40init_hypergate_residualvae_weakaug_BBBC036_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
    },
    'LINCS-Pilot1': {
        'Baseline': ROOT / 'LINCS-Pilot1/followups/E22_lincs_pilot1_followups/formal_train/20260419_210648/train_outputs/main_compare/2026-04-19/MVC_e22_baseline_main_LINCS-Pilot1_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        'MVCPert-HyperGate': ROOT / 'LINCS-Pilot1/followups/E22_lincs_pilot1_followups/formal_train/20260419_210648/train_outputs/main_compare/2026-04-19/MVC_HyperGate_e22_hypergate_main_LINCS-Pilot1_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
        'MVCPert': ROOT / 'LINCS-Pilot1/followups/E40_lincs_pilot1_dose_eval/20260424_1/train_outputs/2026-04-24/MVC_HyperGateResidualVAE_e40init_hypergate_residualvae_doseaware_LINCS-Pilot1_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv',
    },
}

SOTA_RECON = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-09_full_official_v1_augbest_compare/reconstruction_metrics_compare.csv'
SOTA_DOWNSTREAM = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-09_full_official_v1_augbest_compare/full_downstream_compare_table.csv'
SOTA_FIT = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-09_full_official_v1_augbest_compare/fit_pathway_gap_compare_table.csv'
SOTA_SAMPLE = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-09_full_official_v1_augbest_compare/gene_level/bbbc047_gene_level_downstream_sample_metrics.csv'
E40_GENE = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-24_e40_positive_cases/gene_level/bbbc047_gene_level_downstream_summary.csv'
E40_SAMPLE = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-24_e40_positive_cases/gene_level/bbbc047_gene_level_downstream_sample_metrics.csv'
E40_FIT = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-24_e40_positive_cases/fit_pathway_gap/bbbc047_fit_pathway_gap_model_summary.csv'
E40_RETRIEVAL = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-24_e40_positive_cases/retrieval/bbbc047_e40_retrieval_summary.csv'
E40_BBBC036_ENRICH = ROOT / 'BBBC036/followups/E40_bbbc036_same_moa_eval/20260424_1/pair_enrichment/moa_main_top1pct_support5_with_e40.csv'
LINCS_DOSE = ROOT / 'LINCS-Pilot1/followups/E40_lincs_pilot1_dose_eval/20260424_1/dose_aware_retrieval/diagnostic_summary_compare_with_e40.csv'
MORPHDIFF_CP_METRICS = ROOT / 'BBBC047/official_v1/E05_MorphDiff/formal_train/20260410_135409/test_mvc_eval_metrics.csv'
CP_FEATURE_BASELINE_ROOT = ROOT / 'BBBC047/official_v1/E29_CPFeatureBaselines/formal_train'
CP_FEATURE_BASELINE_SPECS = {
    'ResNet-18': {
        'model_key': 'resnet18_feature',
        'family': 'CP feature baseline',
    },
    'Small ViT': {
        'model_key': 'small_vit_feature',
        'family': 'CP feature baseline',
    },
}

ADDITIONAL_BBBC047_SOTA_RUNS = {
    'chemCPA': {
        'model_key': 'chemcpa',
        'root': ROOT / 'BBBC047/official_v1/E01_chemCPA/formal_train/20260410_095432',
    },
    'cycleCDR': {
        'model_key': 'cyclecdr',
        'root': ROOT / 'BBBC047/official_v1/E02_cycleCDR/formal_train/20260410_103004',
    },
    'MiTCP': {
        'model_key': 'mitcp',
        'root': ROOT / 'BBBC047/official_v1/E03_MiTCP/formal_train/20260410_112148',
    },
    'XPert': {
        'model_key': 'xpert',
        'root': ROOT / 'BBBC047/official_v1/E04_XPert/formal_train/20260410_121554',
    },
}

EXTERNAL_SOTA_METRICS = {
    'BBBC036': {
        'PRNet': ROOT / 'BBBC036/followups/E49_panel_cd_external_recon/formal_train/20260516_bbbc036_prnet_pairedfix/prnet_eval/smiles_split_mvc_eval_metrics.csv',
        'TranSiGen': AIDD / 'baseline/Comparision_with_SOTA/Experiment/results/external_sota/transigen_bbbc036_cpu/trained_models_1_cell_smiles_split_3407/feature_ECFP4_init_random/predict/mvc_eval_metrics.csv',
    },
    'LINCS-Pilot1': {
        'PRNet': ROOT / 'LINCS-Pilot1/followups/E49_panel_cd_external_recon/formal_train/20260516_lincs_280_retrain_prnet_pairedfix/aggregated_metrics/prnet_lincs_pilot1_280_mvc_eval_metrics.csv',
    },
}

METHOD_COLORS = {
    'Baseline': '#C96144',
    'MVCPert-HyperGate': '#E99D4E',
    'MVCPert': '#5185C0',
    'PRNet': '#8281B9',
    'TranSiGen': '#55966B',
    'PerturbNet': '#99C290',
    'chemCPA': '#C0BEDC',
    'cycleCDR': '#8EA9D4',
    'MiTCP': '#F2CB9F',
    'XPert': '#99AABB',
    'MorphDiff-like': '#FFB3C1',
    'ResNet-18': '#FFD3E0',
    'Small ViT': '#87CEEB',
    'GE-only target': PALETTE['ge'],
    'CP-only target': PALETTE['cp'],
    'Best GE-only SOTA': '#3A3A3A',
}

METHOD_MARKERS = {
    'Baseline': 'o',
    'MVCPert-HyperGate': 's',
    'MVCPert': 'D',
    'PRNet': '^',
    'TranSiGen': 'v',
    'PerturbNet': 'P',
    'chemCPA': 'o',
    'cycleCDR': '<',
    'MiTCP': '>',
    'XPert': 'X',
    'GE-only target': 'X',
    'CP-only target': 'h',
    'MorphDiff-like': '*',
    'ResNet-18': 'p',
    'Small ViT': 'H',
}

LITERATURE_GE_METHODS = ['PRNet', 'TranSiGen', 'PerturbNet', 'chemCPA', 'cycleCDR', 'MiTCP', 'XPert']
FIG1_SOTA_METHODS = ['MVCPert'] + LITERATURE_GE_METHODS
ALL_METHODS = ['Baseline', 'MVCPert-HyperGate', 'MVCPert'] + LITERATURE_GE_METHODS
MAIN_METHODS = ['Baseline', 'MVCPert-HyperGate', 'MVCPert']
SOTA_METHODS = LITERATURE_GE_METHODS

PANEL_METHOD_LABELS = {
    'Baseline': 'Baseline',
    'MVCPert-HyperGate': 'MVCPert-HG',
    'MVCPert': 'MVCPert',
    'PRNet': 'PRNet',
    'TranSiGen': 'TranSiGen',
    'PerturbNet': 'PerturbNet',
    'chemCPA': 'chemCPA',
    'cycleCDR': 'cycleCDR',
    'MiTCP': 'MiTCP',
    'XPert': 'XPert',
    'MorphDiff-like': 'MorphDiff',
    'ResNet-18': 'ResNet-18',
    'Small ViT': 'Small ViT',
}


def mean_profile(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)

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
        'n': len(df),
    }


def mean_sota_ge_metrics(path: Path) -> dict[str, float]:
    row = pd.read_csv(path).iloc[0]
    return {
        'systema-GE PCC': row.get('systema_GE_PCC', row.get('systema_pearson_GE_mean', np.nan)),
        'GE PCC': row.get('ge_pred_pearson_mean', np.nan),
        'DEG-GE PCC': row.get('DEG_ge_pred_pearson_mean', np.nan),
        'GE RMSE': row.get('ge_pred_rmse_mean', np.nan),
        'DEG-GE RMSE': row.get('DEG_ge_pred_rmse_mean', np.nan),
        'systema-CP PCC': np.nan,
        'CP PCC': np.nan,
        'DEG-CP PCC': np.nan,
        'CP RMSE': np.nan,
        'n': row.get('n_samples', np.nan),
    }


def additional_sota_file(model: str, relative: str) -> Path:
    return ADDITIONAL_BBBC047_SOTA_RUNS[model]['root'] / relative


def latest_cp_feature_baseline_file(model: str, relative: str) -> Path:
    spec = CP_FEATURE_BASELINE_SPECS[model]
    matches = sorted(CP_FEATURE_BASELINE_ROOT.glob(f"*/{spec['model_key']}/{relative}"))
    if not matches:
        raise FileNotFoundError(f"missing CP feature baseline artifact for {model}: {relative}")
    return matches[-1]


def load_structure_ablation_table() -> pd.DataFrame:
    rows = []
    for order, variant in enumerate(STRUCTURE_VARIANT_ORDER):
        metrics = mean_profile(BBBC047_STRUCTURE_PROFILES[variant])
        rows.append(
            {
                'Variant': variant,
                'Display': STRUCTURE_VARIANT_LABELS[variant],
                'Order': order,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def load_structure_step_delta_table(structure: pd.DataFrame | None = None) -> pd.DataFrame:
    if structure is None:
        structure = load_structure_ablation_table()
    keyed = structure.set_index('Variant')
    rows = []
    for order, (step, start, end) in enumerate(STRUCTURE_STEP_DEFINITIONS):
        row = {'Step': step, 'Start': start, 'End': end, 'Order': order}
        for metric in STRUCTURE_PCC_METRICS:
            row[metric] = float(keyed.loc[end, metric] - keyed.loc[start, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def load_augmentation_reconstruction_delta() -> pd.DataFrame:
    raw = pd.read_csv(AUGMENTATION_METRIC_DELTA)
    model_map = {'baseline': 'Baseline', 'dynamics': 'HyperGate'}
    metric_map = {
        'GE systema': 'systema_GE_PCC_delta',
        'GE profile': 'GE_PCC_delta',
        'CP systema': 'systema_CP_PCC_delta',
        'CP profile': 'CP_PCC_delta',
    }
    rows = []
    for _, record in raw.iterrows():
        for metric, column in metric_map.items():
            rows.append(
                {
                    'Model': model_map[record['model']],
                    'Metric': metric,
                    'Delta': float(record[column]),
                }
            )
    return pd.DataFrame(rows)


def load_augmentation_biology_delta() -> pd.DataFrame:
    raw = pd.read_csv(AUGMENTATION_DOWNSTREAM_DELTA)
    model_map = {'baseline': 'Baseline', 'dynamics': 'HyperGate'}
    rows = []
    for model_key, label in model_map.items():
        informative = raw[(raw['model'] == model_key) & (raw['subset'] == 'informative')].copy()
        fit_gap = raw[(raw['model'] == model_key) & (raw['subset'] == 'fit_gap_informative')].copy()
        # In this summary artifact, the fit-gap row reuses sign_delta_vs_noaug to store all-gene PCC delta.
        rows.extend(
            [
                {
                    'Model': label,
                    'Metric': 'Top-50 overlap',
                    'Delta': float(informative.loc[informative['top_k'] == 50.0, 'overlap_delta_vs_noaug'].iloc[0]),
                },
                {
                    'Model': label,
                    'Metric': 'Top-100 overlap',
                    'Delta': float(informative.loc[informative['top_k'] == 100.0, 'overlap_delta_vs_noaug'].iloc[0]),
                },
                {
                    'Model': label,
                    'Metric': 'Sign@100',
                    'Delta': float(informative.loc[informative['top_k'] == 100.0, 'sign_delta_vs_noaug'].iloc[0]),
                },
                {
                    'Model': label,
                    'Metric': 'All-gene PCC',
                    'Delta': float(fit_gap['sign_delta_vs_noaug'].iloc[0]),
                },
            ]
        )
    return pd.DataFrame(rows)


def load_mvcpert_modal_matrix() -> pd.DataFrame:
    rows = []
    label_map = {
        'Dual MVCPert': 'Dual',
        'GE-only target': 'GE-only\ntarget',
        'CP-only target': 'CP-only\ntarget',
    }
    for model, path in BBBC047_CLEAN_UNIMODAL_PROFILES.items():
        metrics = mean_profile(path)
        rows.append(
            {
                'Model': model,
                'Display': label_map[model],
                'systema-GE PCC': metrics['systema-GE PCC'],
                'CP PCC': metrics['CP PCC'],
                'GE PCC': metrics['GE PCC'],
                'systema-CP PCC': metrics['systema-CP PCC'],
                'DEG-CP PCC': metrics['DEG-CP PCC'],
                'DEG-GE PCC': metrics['DEG-GE PCC'],
                'CP RMSE': metrics['CP RMSE'],
                'GE RMSE': metrics['GE RMSE'],
                'n': metrics['n'],
            }
        )
    return pd.DataFrame(rows)


def load_modality_sota_points(recon: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in LITERATURE_GE_METHODS:
        if method in recon.Model.values:
            rows.append(
                {
                    'Endpoint': 'GE PCC',
                    'Model': method,
                    'PCC': float(recon.loc[recon.Model == method, 'GE PCC'].iloc[0]),
                    'Family': 'GE-only SOTA',
                }
            )
    rows.append(
        {
            'Endpoint': 'GE PCC',
            'Model': 'MVCPert',
            'PCC': float(recon.loc[recon.Model == 'MVCPert', 'GE PCC'].iloc[0]),
            'Family': 'MVCPert dual',
        }
    )

    morph = pd.read_csv(MORPHDIFF_CP_METRICS).iloc[0]
    rows.extend(
        [
            {
                'Endpoint': 'CP PCC',
                'Model': 'MorphDiff-like',
                'PCC': float(morph['cp_pred_pearson_mean']),
                'Family': 'CP-only SOTA-like',
            },
            *[
                {
                    'Endpoint': 'CP PCC',
                    'Model': model,
                    'PCC': float(pd.read_csv(latest_cp_feature_baseline_file(model, 'test_mvc_eval_metrics.csv')).iloc[0]['cp_pred_pearson_mean']),
                    'Family': CP_FEATURE_BASELINE_SPECS[model]['family'],
                }
                for model in CP_FEATURE_BASELINE_SPECS
            ],
            {
                'Endpoint': 'CP PCC',
                'Model': 'MVCPert',
                'PCC': float(recon.loc[recon.Model == 'MVCPert', 'CP PCC'].iloc[0]),
                'Family': 'MVCPert dual',
            },
        ]
    )
    return pd.DataFrame(rows)


def annotate_bar_values(
    ax: plt.Axes,
    bars,
    decimals: int = 3,
    pad: float = 0.010,
    fontsize: float = 6.6,
    rotation: float = 0.0,
) -> None:
    for bar in bars:
        height = float(bar.get_height())
        x = bar.get_x() + bar.get_width() / 2
        y = height + pad if height >= 0 else height - pad
        va = 'bottom' if height >= 0 else 'top'
        ax.text(x, y, f'{height:.{decimals}f}', ha='center', va=va, fontsize=fontsize, rotation=rotation)


def load_bbbc047_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, path in BBBC047_PROFILES.items():
        rows.append({'Model': model, 'Type': 'multimodal', **mean_profile(path)})
    recon = pd.DataFrame(rows)
    sota = pd.read_csv(SOTA_RECON)
    for model in ['PRNet', 'TranSiGen', 'PerturbNet']:
        r = sota[sota['model'] == model].iloc[0]
        recon.loc[len(recon)] = {
            'Model': model,
            'Type': 'GE-only',
            'GE PCC': r['GE_PCC'],
            'DEG-GE PCC': r['DEG_GE_PCC'],
            'GE RMSE': r['GE_RMSE'],
            'CP PCC': np.nan,
            'DEG-CP PCC': np.nan,
            'CP RMSE': np.nan,
            'n': r['n_samples'],
        }
    for model in ADDITIONAL_BBBC047_SOTA_RUNS:
        metrics = mean_sota_ge_metrics(additional_sota_file(model, 'test_mvc_eval_metrics.csv'))
        recon.loc[len(recon)] = {'Model': model, 'Type': 'GE-only', **metrics}

    e40_gene = pd.read_csv(E40_GENE)
    down_rows = []
    for subset in ['all', 'informative']:
        for topk in [20, 50, 100]:
            for key, label in [('baseline_multi', 'Baseline'), ('hypergate', 'MVCPert-HyperGate'), ('e40', 'MVCPert')]:
                r = e40_gene[(e40_gene.subset == subset) & (e40_gene.top_k == topk) & (e40_gene.model_key == key)].iloc[0]
                down_rows.append(
                    {'Subset': subset, 'TopK': topk, 'Model': label, 'Overlap': r.mean_overlap_ratio, 'Sign': r.mean_sign_recall}
                )
    sota_ds = pd.read_csv(SOTA_DOWNSTREAM)
    for _, r in sota_ds.iterrows():
        if r['subset'] != 'informative':
            continue
        for model in ['PRNet', 'TranSiGen', 'PerturbNet']:
            down_rows.append(
                {
                    'Subset': 'informative',
                    'TopK': int(r.top_k),
                    'Model': model,
                    'Overlap': r[f'{model}_overlap'],
                    'Sign': r[f'{model}_sign'],
                }
            )
    for model, spec in ADDITIONAL_BBBC047_SOTA_RUNS.items():
        path = additional_sota_file(model, 'downstream/gene_level/bbbc047_gene_level_downstream_summary.csv')
        extra = pd.read_csv(path)
        extra = extra[(extra['subset'] == 'informative') & (extra['model_key'] == spec['model_key'])].copy()
        for _, r in extra.iterrows():
            down_rows.append({'Subset': 'informative', 'TopK': int(r.top_k), 'Model': model, 'Overlap': r.mean_overlap_ratio, 'Sign': r.mean_sign_recall})
    downstream = pd.DataFrame(down_rows)

    e40_fit = pd.read_csv(E40_FIT)
    fit_rows = []
    for key, label in [('baseline_multi', 'Baseline'), ('hypergate', 'MVCPert-HyperGate'), ('e40', 'MVCPert')]:
        r = e40_fit[e40_fit.model_key == key].iloc[0]
        fit_rows.append(
            {
                'Model': label,
                'All-gene PCC': r.mean_all_gene_pcc,
                'Top50 overlap': r.mean_top50_overlap_ratio,
                'Top100 overlap': r.mean_top100_overlap_ratio,
                'Pathway PCC': r.mean_pathway_gene_pcc,
            }
        )
    sota_fit = pd.read_csv(SOTA_FIT)
    for model in ['PRNet', 'TranSiGen', 'PerturbNet']:
        r = sota_fit[sota_fit.model == model].iloc[0]
        fit_rows.append(
            {
                'Model': model,
                'All-gene PCC': r.all_gene_pcc,
                'Top50 overlap': r.top50_overlap,
                'Top100 overlap': r.top100_overlap,
                'Pathway PCC': r.pathway_gene_pcc,
            }
        )
    for model, spec in ADDITIONAL_BBBC047_SOTA_RUNS.items():
        path = additional_sota_file(model, 'downstream/fit_pathway_gap/bbbc047_fit_pathway_gap_model_summary.csv')
        extra = pd.read_csv(path)
        r = extra[extra.model_key == spec['model_key']].iloc[0]
        fit_rows.append(
            {
                'Model': model,
                'All-gene PCC': r.mean_all_gene_pcc,
                'Top50 overlap': r.mean_top50_overlap_ratio,
                'Top100 overlap': r.mean_top100_overlap_ratio,
                'Pathway PCC': r.mean_pathway_gene_pcc,
            }
        )
    fit = pd.DataFrame(fit_rows)

    retrieval = pd.read_csv(E40_RETRIEVAL).assign(
        Model=lambda d: d.model_key.map({'baseline_multi': 'Baseline', 'dynamics_modulation': 'MVCPert-HyperGate', 'e40': 'MVCPert'})
    )
    sample = prepare_sample_advantage()
    return recon, downstream, fit, retrieval, sample


def prepare_sample_advantage(topk: int = 50) -> pd.DataFrame:
    e40 = pd.read_csv(E40_SAMPLE)
    sota = pd.read_csv(SOTA_SAMPLE)
    e40 = e40[(e40.subset == 'informative') & (e40.top_k == topk)].copy()
    sota = sota[(sota.subset == 'informative') & (sota.top_k == topk)].copy()
    keep = ['sample_id', 'smiles', 'moa', 'model_key', 'overlap_ratio', 'sign_recall']
    extra_frames = []
    for model in ADDITIONAL_BBBC047_SOTA_RUNS:
        path = additional_sota_file(model, 'downstream/gene_level/bbbc047_gene_level_downstream_sample_metrics.csv')
        frame = pd.read_csv(path)
        frame = frame[(frame.subset == 'informative') & (frame.top_k == topk)].copy()
        extra_frames.append(frame[keep])
    comb = pd.concat([e40[keep], sota[keep], *extra_frames], ignore_index=True)
    label = {
        'baseline_multi': 'Baseline',
        'hypergate': 'MVCPert-HyperGate',
        'dynamics_modulation': 'MVCPert-HyperGate',
        'e40': 'MVCPert',
        'prnet': 'PRNet',
        'transigen': 'TranSiGen',
        'perturbnet': 'PerturbNet',
        'chemcpa': 'chemCPA',
        'cyclecdr': 'cycleCDR',
        'mitcp': 'MiTCP',
        'xpert': 'XPert',
    }
    comb['Model'] = comb.model_key.map(label)
    piv = comb.pivot_table(index=['sample_id', 'smiles', 'moa'], columns='Model', values='overlap_ratio', aggfunc='mean').reset_index()
    for m in ALL_METHODS:
        if m not in piv:
            piv[m] = np.nan
    piv['Best GE-only SOTA'] = piv[SOTA_METHODS].max(axis=1)
    piv['Best SOTA method'] = piv[SOTA_METHODS].idxmax(axis=1)
    piv['MVCPert minus Baseline'] = piv['MVCPert'] - piv['Baseline']
    piv['MVCPert minus best GE-only'] = piv['MVCPert'] - piv['Best GE-only SOTA']
    piv['MVCPert minus best literature'] = piv['MVCPert minus best GE-only']
    piv['MVCPert-HG minus Baseline'] = piv['MVCPert-HyperGate'] - piv['Baseline']
    return piv


def add_legend(fig, methods=ALL_METHODS, y=0.94):
    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS.get(m, 'o'),
            color='none',
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor='white',
            markersize=8,
            label=m,
        )
        for m in methods
    ]
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, y),
        ncol=min(len(methods), 6),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )


def init_plot_env() -> None:
    apply_publication_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
