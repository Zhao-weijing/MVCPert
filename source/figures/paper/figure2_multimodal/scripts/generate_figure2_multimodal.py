#!/usr/bin/env python3
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPT_DIR.parents[1] / 'shared'
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

ARTIFACT_ROOT = Path(os.environ.get("MVCPERT_ARTIFACT_ROOT", "<ARTIFACT_ROOT>"))
USE_LOCAL_ANALYSIS = os.environ.get("MVCPERT_USE_LOCAL_ANALYSIS", "1").lower() not in {"0", "false", "no"}

from paper_plot_style import ANALYSIS_DIR, PALETTE, clean_axes, save_figure
from paper_figure_data_common import (
    AIDD,
    ADDITIONAL_BBBC047_SOTA_RUNS,
    BBBC047_CLEAN_UNIMODAL_PROFILES,
    BBBC047_PROFILES,
    E40_SAMPLE,
    FIG1_SOTA_METHODS,
    LITERATURE_GE_METHODS,
    METHOD_COLORS,
    METHOD_MARKERS,
    PANEL_METHOD_LABELS,
    SOTA_SAMPLE,
    additional_sota_file,
    init_plot_env,
    latest_cp_feature_baseline_file,
    load_bbbc047_tables,
    load_modality_sota_points,
)


BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 20260509
SEED_ORDER = [3407, 0, 10, 100, 1000]
PANEL_C_SEED_METRICS = ANALYSIS_DIR / 'figure2_panel_c_multiseed_seed_metrics_used.csv'
BBBC036_EXTERNAL_SEED_METRICS = AIDD / 'baseline/docs/reports/figures/2026-05-16-panel-cd-external-recon-materials/panel_cd_external_recon_seed_metrics.csv'
FIG1_EXCLUDED_METHODS = {'PerturbNet'}
FIG1_GE_METHODS = [method for method in FIG1_SOTA_METHODS if method not in FIG1_EXCLUDED_METHODS]
FIG1_LITERATURE_GE_METHODS = [method for method in LITERATURE_GE_METHODS if method not in FIG1_EXCLUDED_METHODS]
LEGACY_PRIMARY_NAME = 'Trans' + 'Morph'
LEGACY_HYPERGATE_NAME = 'TM-' + 'HyperGate'
LEGACY_LABEL_MAP = {
    LEGACY_PRIMARY_NAME: 'MVCPert',
    LEGACY_HYPERGATE_NAME: 'MVCPert-HyperGate',
}

OLD_FIT_SAMPLE = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-09_full_official_v1_augbest_compare/fit_pathway_gap/bbbc047_fit_pathway_gap_sample_metrics.csv'
E40_FIT_SAMPLE = AIDD / 'baseline/pathway_benchmark/outputs/2026-04-24_e40_positive_cases/fit_pathway_gap/bbbc047_fit_pathway_gap_sample_metrics.csv'
PRNET_RESULTS_DIR = AIDD / 'baseline/Comparision_with_SOTA/Experiment/results/official_v1_retrain/prnet'
PRNET_H5AD = AIDD / 'baseline/Comparision_with_SOTA/Experiment/datasets/bbbc047_prnet_officialv1_one_sample.h5ad'
PRNET_FIXED_REFRESH_ROOT = Path(os.environ.get(
    "MVCPERT_FIG2_PRNET_REFRESH_ROOT",
    str(ARTIFACT_ROOT / "BBBC047/followups/E48_prnet_fixed_figure1_refresh/20260515_232940"),
))
PRNET_FIXED_GENE_SAMPLE = PRNET_FIXED_REFRESH_ROOT / 'gene_level' / 'bbbc047_gene_level_downstream_sample_metrics.csv'
PRNET_FIXED_GENE_SUMMARY = PRNET_FIXED_REFRESH_ROOT / 'gene_level' / 'bbbc047_gene_level_downstream_summary.csv'
PRNET_FIXED_FIT_SAMPLE = PRNET_FIXED_REFRESH_ROOT / 'fit_pathway_gap' / 'bbbc047_fit_pathway_gap_sample_metrics.csv'
PRNET_FIXED_FIT_SUMMARY = PRNET_FIXED_REFRESH_ROOT / 'fit_pathway_gap' / 'bbbc047_fit_pathway_gap_model_summary.csv'
PERTURBNET_RESULTS_DIR = AIDD / 'baseline/Comparision_with_SOTA/Experiment/results/official_v1_retrain/perturbnet/predictions'
TRANSIGEN_SAMPLE_CSV = AIDD / 'baseline/Comparision_with_SOTA/Experiment/results/official_v1_retrain/transigen_retry/trained_models_1_cell_smiles_split_3407/feature_ECFP4_init_random/predict/test_reconstruction_result_all_samples.csv'
MORPHDIFF_PROFILE = Path(os.environ.get(
    "MVCPERT_FIG2_MORPHDIFF_PROFILE",
    str(ARTIFACT_ROOT / "BBBC047/official_v1/E05_MorphDiff/formal_train/20260410_135409/profiles/morphdiff_test_prediction_profile.h5"),
))
UNIMODAL_MULTISEED_ROOT = Path(os.environ.get(
    "MVCPERT_FIG2_UNIMODAL_MULTISEED_ROOT",
    str(ARTIFACT_ROOT / "BBBC047/followups/E44_bbbc047_clean_unimodal_multiseed_ci/formal_train"),
))
MVCPERT_MULTISEED_ROOT = Path(os.environ.get(
    "MVCPERT_FIG2_MVCPERT_MULTISEED_ROOT",
    str(ARTIFACT_ROOT / "BBBC047/followups/E42_bbbc047_transmorph_variance_controls/formal_train"),
))
DUAL_VS_GE_ONLY_PROBE_ROOT = Path(os.environ.get(
    "MVCPERT_FIG2_DUAL_VS_GE_ONLY_ROOT",
    str(ARTIFACT_ROOT / "BBBC047/followups/E46_bbbc047_dual_vs_ge_only_downstream_probe/formal_eval"),
))

MODEL_KEY_TO_LABEL = {
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

UNIMODAL_MULTISEED_SPECS = {
    'Dual MVCPert': {
        3407: BBBC047_CLEAN_UNIMODAL_PROFILES['Dual MVCPert'],
        0: ('MVC_HyperGateResidualVAE', 'e42_transmorph_corrected_main_seed0'),
        10: ('MVC_HyperGateResidualVAE', 'e42_transmorph_corrected_main_seed10'),
        100: ('MVC_HyperGateResidualVAE', 'e42_transmorph_corrected_main_seed100'),
        1000: ('MVC_HyperGateResidualVAE', 'e42_transmorph_corrected_main_seed1000'),
    },
    'GE-only target': {
        3407: BBBC047_CLEAN_UNIMODAL_PROFILES['GE-only target'],
        0: ('MVC_HyperGateResidualVAE_NoCP', 'e44_clean_ge_only_target_seed0'),
        10: ('MVC_HyperGateResidualVAE_NoCP', 'e44_clean_ge_only_target_seed10'),
        100: ('MVC_HyperGateResidualVAE_NoCP', 'e44_clean_ge_only_target_seed100'),
        1000: ('MVC_HyperGateResidualVAE_NoCP', 'e44_clean_ge_only_target_seed1000'),
    },
    'CP-only target': {
        3407: BBBC047_CLEAN_UNIMODAL_PROFILES['CP-only target'],
        0: ('MVC_HyperGateResidualVAE_NoGE', 'e44_clean_cp_only_target_seed0'),
        10: ('MVC_HyperGateResidualVAE_NoGE', 'e44_clean_cp_only_target_seed10'),
        100: ('MVC_HyperGateResidualVAE_NoGE', 'e44_clean_cp_only_target_seed100'),
        1000: ('MVC_HyperGateResidualVAE_NoGE', 'e44_clean_cp_only_target_seed1000'),
    },
}


def _decode_text(values) -> np.ndarray:
    return np.array(
        [v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v) for v in values],
        dtype=object,
    )


def _decode_cat(node) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        return _decode_text(node[:])
    categories = _decode_text(node['categories'][:])
    codes = np.asarray(node['codes'][:], dtype=np.int64)
    return categories[codes]


def _rowwise_pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pred = pred - pred.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    denom = np.sqrt((pred * pred).sum(axis=1) * (target * target).sum(axis=1))
    return np.divide((pred * target).sum(axis=1), denom, out=np.zeros_like(denom), where=denom > 0)


def _collapse_metric_frame(df: pd.DataFrame, key: str, metric: str) -> pd.DataFrame:
    frame = df[[key, metric]].dropna().copy()
    frame[key] = frame[key].astype(str)
    frame = frame.groupby(key, as_index=False)[metric].mean()
    return frame


@lru_cache(maxsize=16)
def _bootstrap_indices(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED + n * 17)
    return rng.integers(0, n, size=(BOOTSTRAP_ITERS, n), dtype=np.int32)


def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean, 1
    boot = arr[_bootstrap_indices(n)].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), n


def _aligned_bootstrap_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    left_metric: str,
    right_metric: str,
) -> tuple[float, float, float, int]:
    merged = left.merge(right, on=key, how='inner', suffixes=('_left', '_right'))
    delta = merged[f'{right_metric}_right'].to_numpy(dtype=np.float64) - merged[f'{left_metric}_left'].to_numpy(dtype=np.float64)
    return _bootstrap_mean_ci(delta)


def _latest_multiseed_predict_path(model_type: str, output_tag: str) -> Path:
    roots = [UNIMODAL_MULTISEED_ROOT, MVCPERT_MULTISEED_ROOT]
    pattern = f'*/train_outputs/*/{model_type}_{output_tag}_BBBC047_smiles_split/ECFP4_Default/predict/test_restruction_result_all_samples.csv'
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(sorted(root.glob(pattern)))
    if not matches:
        raise FileNotFoundError(f'no multiseed result found for {model_type} / {output_tag}')
    return matches[-1]


def _unimodal_seed_predict_path(model: str, seed: int) -> Path:
    spec = UNIMODAL_MULTISEED_SPECS[model][seed]
    return _latest_multiseed_predict_path(*spec) if isinstance(spec, tuple) else Path(spec)


def _load_profile_metric_frame(path: Path, pred_key: str, target_key: str, metric_name: str) -> pd.DataFrame:
    with h5py.File(path, 'r') as handle:
        smiles = _decode_text(handle['smiles'][:])
        pred = np.asarray(handle[pred_key][:], dtype=np.float32)
        target = np.asarray(handle[target_key][:], dtype=np.float32)
    metric = _rowwise_pearson(pred, target)
    return _collapse_metric_frame(pd.DataFrame({'canonical_smiles': smiles, metric_name: metric}), 'canonical_smiles', metric_name)


@lru_cache(maxsize=1)
def _load_prnet_ge_frame() -> pd.DataFrame:
    pred = pd.read_csv(PRNET_RESULTS_DIR / 'smiles_split_y_pre_array.csv', header=None).to_numpy(dtype=np.float32)
    true = pd.read_csv(PRNET_RESULTS_DIR / 'smiles_split_y_true_array.csv', header=None).to_numpy(dtype=np.float32)
    cov_names = pd.read_csv(PRNET_RESULTS_DIR / 'smiles_split_cov_drug_array.csv', header=None)[0].astype(str).to_numpy()
    with h5py.File(PRNET_H5AD, 'r') as handle:
        obs = handle['obs']
        smiles = _decode_cat(obs['SMILES'])
        split = _decode_cat(obs['smiles_split'])
        cov = _decode_cat(obs['cov_drug_name'])
    test_df = pd.DataFrame({'canonical_smiles': smiles, 'split': split, 'cov': cov})
    test_df = test_df[test_df['split'] == 'test'][['canonical_smiles', 'cov']].reset_index(drop=True)
    cov_to_smiles = dict(zip(test_df['cov'].astype(str), test_df['canonical_smiles'].astype(str)))
    aligned_smiles = np.array([cov_to_smiles[name] for name in cov_names if name in cov_to_smiles], dtype=object)
    if len(aligned_smiles) != len(pred):
        raise ValueError(f'PRNet alignment mismatch: smiles={len(aligned_smiles)} pred={len(pred)}')
    metric = _rowwise_pearson(pred, true)
    return _collapse_metric_frame(pd.DataFrame({'canonical_smiles': aligned_smiles, 'ge_pred_pearson': metric}), 'canonical_smiles', 'ge_pred_pearson')


@lru_cache(maxsize=1)
def _load_perturbnet_ge_frame() -> pd.DataFrame:
    smiles = np.load(PERTURBNET_RESULTS_DIR / 'test_smiles.npy', allow_pickle=True)
    smiles = np.array([s[0] if isinstance(s, (list, tuple, np.ndarray)) else s for s in smiles], dtype=object)
    pred = np.load(PERTURBNET_RESULTS_DIR / 'test_predictions.npy', allow_pickle=True).astype(np.float32)
    true = np.load(PERTURBNET_RESULTS_DIR / 'test_ground_truth.npy', allow_pickle=True).astype(np.float32)
    metric = _rowwise_pearson(pred, true)
    return _collapse_metric_frame(pd.DataFrame({'canonical_smiles': smiles.astype(str), 'ge_pred_pearson': metric}), 'canonical_smiles', 'ge_pred_pearson')


@lru_cache(maxsize=1)
def _load_transigen_ge_frame() -> pd.DataFrame:
    df = pd.read_csv(TRANSIGEN_SAMPLE_CSV)
    frame = df[['canonical_smiles', 'x2_pred_pearson']].rename(columns={'x2_pred_pearson': 'ge_pred_pearson'})
    return _collapse_metric_frame(frame, 'canonical_smiles', 'ge_pred_pearson')


@lru_cache(maxsize=8)
def _load_additional_ge_frame(model: str) -> pd.DataFrame:
    spec = ADDITIONAL_BBBC047_SOTA_RUNS[model]
    path = spec['root'] / 'profiles' / f"{spec['model_key']}_test_prediction_profile.h5"
    return _load_profile_metric_frame(path, pred_key='ge_pred', target_key='target_ge', metric_name='ge_pred_pearson')


@lru_cache(maxsize=1)
def _load_morphdiff_cp_frame() -> pd.DataFrame:
    return _load_profile_metric_frame(MORPHDIFF_PROFILE, pred_key='cp_pred', target_key='target_cp', metric_name='cp_pred_pearson')


def _load_csv_metric_frame(path: Path, metric_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _collapse_metric_frame(df[['canonical_smiles', metric_name]], 'canonical_smiles', metric_name)


@lru_cache(maxsize=1)
def _load_prnet_fixed_downstream_summary() -> pd.DataFrame:
    return pd.read_csv(PRNET_FIXED_GENE_SUMMARY)


@lru_cache(maxsize=1)
def _load_prnet_fixed_fit_summary() -> pd.DataFrame:
    return pd.read_csv(PRNET_FIXED_FIT_SUMMARY)


def load_ge_reconstruction_frame(method: str) -> pd.DataFrame:
    if method == 'MVCPert':
        return _load_csv_metric_frame(BBBC047_PROFILES['MVCPert'], 'ge_pred_pearson').copy()
    if method == 'PRNet':
        return _load_prnet_ge_frame().copy()
    if method == 'TranSiGen':
        return _load_transigen_ge_frame().copy()
    if method == 'PerturbNet':
        return _load_perturbnet_ge_frame().copy()
    if method in ADDITIONAL_BBBC047_SOTA_RUNS:
        return _load_additional_ge_frame(method).copy()
    raise KeyError(f'unsupported GE reconstruction method: {method}')


def load_cp_reconstruction_frame(method: str) -> pd.DataFrame:
    if method == 'MVCPert':
        return _load_csv_metric_frame(BBBC047_PROFILES['MVCPert'], 'cp_pred_pearson').copy()
    if method == 'MorphDiff-like':
        return _load_morphdiff_cp_frame().copy()
    if method in {'ResNet-18', 'Small ViT'}:
        profile = latest_cp_feature_baseline_file(method, 'profiles/' + {'ResNet-18': 'resnet18_feature', 'Small ViT': 'small_vit_feature'}[method] + '_test_prediction_profile.h5')
        return _load_profile_metric_frame(profile, pred_key='cp_pred', target_key='target_cp', metric_name='cp_pred_pearson')
    raise KeyError(f'unsupported CP reconstruction method: {method}')


def load_unimodal_metric_frame(model: str, metric_name: str) -> pd.DataFrame:
    return _load_csv_metric_frame(BBBC047_CLEAN_UNIMODAL_PROFILES[model], metric_name).copy()


def load_downstream_overlap_frame(method: str, topk: int) -> pd.DataFrame:
    if method == 'MVCPert':
        path = E40_SAMPLE
        model_key = 'e40'
    elif method == 'PRNet':
        path = PRNET_FIXED_GENE_SAMPLE
        model_key = 'prnet'
    elif method in {'TranSiGen', 'PerturbNet'}:
        path = SOTA_SAMPLE
        model_key = method.lower()
    elif method in ADDITIONAL_BBBC047_SOTA_RUNS:
        path = additional_sota_file(method, 'downstream/gene_level/bbbc047_gene_level_downstream_sample_metrics.csv')
        model_key = ADDITIONAL_BBBC047_SOTA_RUNS[method]['model_key']
    else:
        raise KeyError(f'unsupported downstream method: {method}')
    df = pd.read_csv(path)
    sub = df[(df['subset'] == 'informative') & (df['top_k'] == topk) & (df['model_key'] == model_key)].copy()
    return sub[['sample_id', 'overlap_ratio']].dropna().groupby('sample_id', as_index=False)['overlap_ratio'].mean()


def load_fit_metric_frame(method: str, metric_name: str) -> pd.DataFrame:
    if method == 'MVCPert':
        path = E40_FIT_SAMPLE
        model_key = 'e40'
    elif method == 'PRNet':
        path = PRNET_FIXED_FIT_SAMPLE
        model_key = 'prnet'
    elif method in {'TranSiGen', 'PerturbNet'}:
        path = OLD_FIT_SAMPLE
        model_key = method.lower()
    elif method in ADDITIONAL_BBBC047_SOTA_RUNS:
        path = additional_sota_file(method, 'downstream/fit_pathway_gap/bbbc047_fit_pathway_gap_sample_metrics.csv')
        model_key = ADDITIONAL_BBBC047_SOTA_RUNS[method]['model_key']
    else:
        raise KeyError(f'unsupported fit/pathway method: {method}')
    df = pd.read_csv(path)
    sub = df[(df['is_informative'] == 1) & (df['model_key'] == model_key)].copy()
    return sub[['sample_id', metric_name]].dropna().groupby('sample_id', as_index=False)[metric_name].mean()


def compute_modal_gain_error_table(
    endpoint_metrics: list[tuple[str, str, str, str, str]],
    higher_is_better: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    seed_rows = []
    use_seed_ci = True
    for model in ['Dual MVCPert', 'GE-only target', 'CP-only target']:
        for seed in SEED_ORDER:
            try:
                path = _unimodal_seed_predict_path(model, seed)
            except FileNotFoundError:
                use_seed_ci = False
                break
            df = pd.read_csv(path)
            row = {'Model': model, 'Seed': seed}
            for metric_key, column, _, _, _ in endpoint_metrics:
                row[metric_key] = float(df[column].mean()) if column in df.columns else np.nan
            seed_rows.append(row)
        if not use_seed_ci:
            break
    seed_table = pd.DataFrame(seed_rows) if use_seed_ci else None

    rows = []
    if use_seed_ci and seed_table is not None:
        dual = seed_table[seed_table['Model'] == 'Dual MVCPert'].set_index('Seed')
        for metric_key, _, label, control, color in endpoint_metrics:
            control_table = seed_table[seed_table['Model'] == control].set_index('Seed')
            dual_vals = dual.loc[SEED_ORDER, metric_key].to_numpy(dtype=np.float64)
            control_vals = control_table.loc[SEED_ORDER, metric_key].to_numpy(dtype=np.float64)
            delta = dual_vals - control_vals if higher_is_better else control_vals - dual_vals
            delta_mean, ci_low, ci_high, n = _bootstrap_mean_ci(delta)
            rows.append(
                {
                    'MetricKey': metric_key,
                    'Metric': label,
                    'Control': control,
                    'Color': color,
                    'Clean': float(control_table.loc[SEED_ORDER, metric_key].mean()),
                    'Dual': float(dual.loc[SEED_ORDER, metric_key].mean()),
                    'Delta': delta_mean,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'n': n,
                    'error_source': 'seed_bootstrap',
                }
            )
    else:
        for metric_key, column, label, control, color in endpoint_metrics:
            dual_df = load_unimodal_metric_frame('Dual MVCPert', column)
            control_df = load_unimodal_metric_frame(control, column)
            if higher_is_better:
                delta_mean, ci_low, ci_high, n = _aligned_bootstrap_delta(control_df, dual_df, 'canonical_smiles', column, column)
            else:
                delta_mean, ci_low, ci_high, n = _aligned_bootstrap_delta(dual_df, control_df, 'canonical_smiles', column, column)
            rows.append(
                {
                    'MetricKey': metric_key,
                    'Metric': label,
                    'Control': control,
                    'Color': color,
                    'Clean': float(control_df[column].mean()),
                    'Dual': float(dual_df[column].mean()),
                    'Delta': delta_mean,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'n': n,
                    'error_source': 'paired_sample_bootstrap',
                }
            )
    summary = pd.DataFrame(rows)
    for col in ['Delta', 'ci_low', 'ci_high']:
        summary[f'{col}_x1e3'] = summary[col] * 1000.0
    return summary, seed_table


def compute_modality_method_error_table(recon: pd.DataFrame) -> pd.DataFrame:
    point_df = load_modality_sota_points(recon)
    rows = []
    ge_methods = point_df[point_df['Endpoint'] == 'GE PCC']['Model'].tolist()
    for method in ge_methods:
        frame = load_ge_reconstruction_frame(method)
        mean, ci_low, ci_high, n = _bootstrap_mean_ci(frame['ge_pred_pearson'].to_numpy())
        rows.append({'Endpoint': 'GE PCC', 'Model': method, 'PCC': mean, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n, 'error_source': 'sample_bootstrap'})
    cp_methods = point_df[point_df['Endpoint'] == 'CP PCC']['Model'].tolist()
    for method in cp_methods:
        frame = load_cp_reconstruction_frame(method)
        mean, ci_low, ci_high, n = _bootstrap_mean_ci(frame['cp_pred_pearson'].to_numpy())
        rows.append({'Endpoint': 'CP PCC', 'Model': method, 'PCC': mean, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n, 'error_source': 'sample_bootstrap'})
    return point_df.drop(columns=['PCC']).merge(pd.DataFrame(rows), on=['Endpoint', 'Model'], how='left')


def load_panel_c_seed_metrics() -> pd.DataFrame:
    if PANEL_C_SEED_METRICS.exists():
        df = pd.read_csv(PANEL_C_SEED_METRICS)
        keep = df[df['Endpoint'].isin({'GE PCC', 'CP PCC'}) & df['Model'].notna()].copy()
        keep['Model'] = keep['Model'].replace(LEGACY_LABEL_MAP)
        keep = keep[~keep['Model'].isin(FIG1_EXCLUDED_METHODS)].copy()
        keep['Seed'] = keep['Seed'].astype(int)
        keep['PCC'] = keep['PCC'].astype(float)
        return keep

    rows = []
    for method in FIG1_GE_METHODS:
        frame = load_ge_reconstruction_frame(method)
        rows.append({'Endpoint': 'GE PCC', 'Model': method, 'Seed': 0, 'PCC': float(frame['ge_pred_pearson'].mean())})
    for method in ['MVCPert', 'MorphDiff-like', 'ResNet-18', 'Small ViT']:
        frame = load_cp_reconstruction_frame(method)
        rows.append({'Endpoint': 'CP PCC', 'Model': method, 'Seed': 0, 'PCC': float(frame['cp_pred_pearson'].mean())})
    return pd.DataFrame(rows)


def summarize_panel_c_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(['Endpoint', 'Model'], as_index=False)
        .agg(mean_pcc=('PCC', 'mean'), std_pcc=('PCC', 'std'), n=('PCC', 'size'))
        .sort_values(['Endpoint', 'mean_pcc'], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary['std_pcc'] = summary['std_pcc'].fillna(0.0)
    return summary


def load_bbbc036_external_deg_seed_metrics() -> pd.DataFrame:
    raw = pd.read_csv(BBBC036_EXTERNAL_SEED_METRICS)
    raw = raw[(raw['dataset'] == 'BBBC036') & (raw['status'] == 'completed')].copy()

    rows = []
    ge_rows = raw[(raw['panel'] == 'C') & raw['deg_ge_pcc'].notna()].copy()
    for row in ge_rows.itertuples(index=False):
        rows.append(
            {
                'Endpoint': 'DEG-GE PCC',
                'Model': LEGACY_LABEL_MAP.get(str(row.method), str(row.method)),
                'Seed': int(row.seed),
                'PCC': float(row.deg_ge_pcc),
            }
        )

    cp_rows = raw[(raw['panel'] == 'D') & raw['deg_cp_pcc'].notna()].copy()
    for row in cp_rows.itertuples(index=False):
        rows.append(
            {
                'Endpoint': 'DEG-CP PCC',
                'Model': LEGACY_LABEL_MAP.get(str(row.method), str(row.method)),
                'Seed': int(row.seed),
                'PCC': float(row.deg_cp_pcc),
            }
        )

    return pd.DataFrame(rows)


def _panel_c_row_style(model: str, best_baseline: str | None) -> tuple[str, float, float]:
    del best_baseline
    if model == 'MVCPert':
        return METHOD_COLORS['MVCPert'], 1.0, 1.0
    return METHOD_COLORS.get(model, '#B8B1C4'), 0.92, 0.92


def draw_panel_c_interval_row(
    ax: plt.Axes,
    values: np.ndarray,
    pos: float,
    color: str,
    marker: str,
    rng: np.random.Generator,
    alpha: float,
    marker_alpha: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    vmin = float(values.min())
    vmax = float(values.max())

    ax.hlines(pos, vmin, vmax, color=color, lw=1.0, alpha=0.28 + 0.38 * alpha, zorder=1)
    ax.hlines(pos, mean - std, mean + std, color=color, lw=3.0, alpha=alpha, zorder=2)
    ax.vlines([mean - std, mean + std], pos - 0.08, pos + 0.08, color=color, lw=1.0, alpha=alpha * 0.85, zorder=2)

    jitter = rng.normal(loc=0.0, scale=0.035, size=values.size)
    ax.scatter(
        values,
        pos + jitter,
        s=28,
        marker=marker,
        color=color,
        alpha=marker_alpha,
        edgecolor='white',
        linewidth=0.65,
        zorder=3,
    )
    ax.scatter(
        [mean],
        [pos],
        marker=marker,
        s=48,
        color=color,
        edgecolor='white',
        linewidth=0.75,
        zorder=4,
    )
    return mean, std


def plot_panel_c_bar_subset(
    ax: plt.Axes,
    panel_c: pd.DataFrame,
    endpoint: str,
    title: str,
    min_span: float,
    xlabel: str = 'PCC',
) -> None:
    subset = panel_c[panel_c['Endpoint'] == endpoint].copy()
    summary = (
        subset.groupby('Model', as_index=False)
        .agg(mean_pcc=('PCC', 'mean'), std_pcc=('PCC', 'std'))
        .sort_values('mean_pcc', ascending=False)
        .reset_index(drop=True)
    )
    summary['std_pcc'] = summary['std_pcc'].fillna(0.0)
    order = summary['Model'].tolist()
    best_baseline = next((model for model in order if model != 'MVCPert'), None)
    y_positions = np.arange(len(order))[::-1]
    labels: list[str] = []

    for idx, model in enumerate(order):
        pos = float(y_positions[idx])
        row = summary.iloc[idx]
        color, alpha, _ = _panel_c_row_style(str(model), best_baseline)
        mean = float(row['mean_pcc'])
        std = float(row['std_pcc'])
        ax.barh(
            pos,
            mean,
            xerr=std,
            color=color,
            alpha=alpha,
            height=0.62,
            edgecolor='white',
            linewidth=0.8,
            error_kw={
                'ecolor': color,
                'elinewidth': 1.8,
                'capsize': 3.2,
                'capthick': 1.4,
            },
            zorder=3,
        )
        labels.append(PANEL_METHOD_LABELS.get(str(model), str(model)))

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel(xlabel, fontsize=11.5)
    raw_min = float((summary['mean_pcc'] - summary['std_pcc']).min())
    raw_max = float((summary['mean_pcc'] + summary['std_pcc']).max())
    span = max(raw_max - raw_min, min_span)
    xpad_left = max(0.008, 0.08 * span)
    xpad_right = max(0.014, 0.14 * span)
    xmin = max(0.0, raw_min - xpad_left)
    xmax = raw_max + xpad_right
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.50, float(len(order) - 0.25))
    ax.set_title(title, fontsize=13.8, pad=8)
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', labelsize=10.5)
    clean_axes(ax)


def compute_recovery_error_table() -> pd.DataFrame:
    rows = []
    for method in FIG1_GE_METHODS:
        for topk in [20, 50, 100]:
            frame = load_downstream_overlap_frame(method, topk)
            mean, ci_low, ci_high, n = _bootstrap_mean_ci(frame['overlap_ratio'].to_numpy())
            rows.append(
                {
                    'Model': method,
                    'TopK': topk,
                    'Subset': 'informative',
                    'Overlap': mean,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'n': n,
                    'error_source': 'sample_bootstrap',
                }
            )
    return pd.DataFrame(rows)


def compute_margin_error_table(method_ci: pd.DataFrame, recovery_ci: pd.DataFrame) -> pd.DataFrame:
    ge_best = method_ci[(method_ci['Endpoint'] == 'GE PCC') & (method_ci['Model'] != 'MVCPert')].sort_values('PCC', ascending=False).iloc[0]
    cp_best = method_ci[(method_ci['Endpoint'] == 'CP PCC') & (method_ci['Model'] != 'MVCPert')].sort_values('PCC', ascending=False).iloc[0]
    top50_best = recovery_ci[(recovery_ci['Model'].isin(FIG1_LITERATURE_GE_METHODS)) & (recovery_ci['TopK'] == 50)].sort_values('Overlap', ascending=False).iloc[0]
    top100_best = recovery_ci[(recovery_ci['Model'].isin(FIG1_LITERATURE_GE_METHODS)) & (recovery_ci['TopK'] == 100)].sort_values('Overlap', ascending=False).iloc[0]

    rows = []

    tm_ge = load_ge_reconstruction_frame('MVCPert')
    comp_ge = load_ge_reconstruction_frame(str(ge_best['Model']))
    gain, ci_low, ci_high, n = _aligned_bootstrap_delta(comp_ge, tm_ge, 'canonical_smiles', 'ge_pred_pearson', 'ge_pred_pearson')
    rows.append(
        {
            'Metric': 'GE prediction',
            'Gain': gain * 100.0,
            'ci_low': ci_low * 100.0,
            'ci_high': ci_high * 100.0,
            'Comparator': str(ge_best['Model']),
            'Color': PALETTE['ge'],
            'n': n,
            'error_source': 'paired_sample_bootstrap',
        }
    )

    tm_cp = load_cp_reconstruction_frame('MVCPert')
    comp_cp = load_cp_reconstruction_frame(str(cp_best['Model']))
    gain, ci_low, ci_high, n = _aligned_bootstrap_delta(comp_cp, tm_cp, 'canonical_smiles', 'cp_pred_pearson', 'cp_pred_pearson')
    rows.append(
        {
            'Metric': 'CP prediction',
            'Gain': gain * 100.0,
            'ci_low': ci_low * 100.0,
            'ci_high': ci_high * 100.0,
            'Comparator': str(cp_best['Model']),
            'Color': PALETTE['cp'],
            'n': n,
            'error_source': 'paired_sample_bootstrap',
        }
    )

    for topk, label, comparator in [(50, 'Top-50 recovery', str(top50_best['Model'])), (100, 'Top-100 recovery', str(top100_best['Model']))]:
        tm_ds = load_downstream_overlap_frame('MVCPert', topk)
        comp_ds = load_downstream_overlap_frame(comparator, topk)
        gain, ci_low, ci_high, n = _aligned_bootstrap_delta(comp_ds, tm_ds, 'sample_id', 'overlap_ratio', 'overlap_ratio')
        rows.append(
            {
                'Metric': label,
                'Gain': gain * 100.0,
                'ci_low': ci_low * 100.0,
                'ci_high': ci_high * 100.0,
                'Comparator': comparator,
                'Color': METHOD_COLORS['MVCPert'],
                'n': n,
                'error_source': 'paired_sample_bootstrap',
            }
        )
    return pd.DataFrame(rows)


def _latest_dual_vs_ge_only_probe_file(relative: str) -> Path:
    matches = sorted(DUAL_VS_GE_ONLY_PROBE_ROOT.glob(f'*/{relative}'))
    if not matches:
        raise FileNotFoundError(f'no dual-vs-ge-only probe file found for {relative}')
    return matches[-1]


def load_ge_only_downstream_gain_table() -> pd.DataFrame:
    summary = pd.read_csv(_latest_dual_vs_ge_only_probe_file('gene_level/bbbc047_gene_level_downstream_summary.csv'))
    metric_specs = [
        ('Overlap', 20, 'mean_overlap_ratio', 'Informative overlap@20', PALETTE['dual']),
        ('Overlap', 50, 'mean_overlap_ratio', 'Informative overlap@50', PALETTE['dual']),
        ('Overlap', 100, 'mean_overlap_ratio', 'Informative overlap@100', PALETTE['dual']),
        ('Sign recall', 20, 'mean_sign_recall', 'Top20 sign recall', PALETTE['ge']),
        ('Sign recall', 50, 'mean_sign_recall', 'Top50 sign recall', PALETTE['ge']),
        ('Sign recall', 100, 'mean_sign_recall', 'Top100 sign recall', PALETTE['ge']),
    ]
    info = summary[summary['subset'] == 'informative'].copy()
    rows = []
    for group, topk, column, label, color in metric_specs:
        sub = info[info['top_k'] == topk].set_index('model_key')
        dual = float(sub.loc['dual', column])
        ge_only = float(sub.loc['ge_only', column])
        rows.append(
            {
                'Group': group,
                'TopK': topk,
                'Metric': label,
                'Dual': dual,
                'GE-only': ge_only,
                'Delta': dual - ge_only,
                'Delta_pct': (dual - ge_only) * 100.0,
                'Color': color,
            }
        )
    return pd.DataFrame(rows)


def _replace_prnet_rows(
    recon: pd.DataFrame,
    downstream: pd.DataFrame,
    fit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prnet_recon = pd.read_csv(PRNET_RESULTS_DIR / 'smiles_split_mvc_eval_metrics.csv').iloc[0]
    recon_row = recon[recon['Model'] == 'PRNet'].iloc[0].to_dict() if (recon['Model'] == 'PRNet').any() else {'Model': 'PRNet', 'Type': 'GE-only'}
    recon_row.update(
        {
            'Model': 'PRNet',
            'Type': 'GE-only',
            'systema-GE PCC': float(prnet_recon.get('systema_GE_PCC', prnet_recon.get('systema_pearson_GE_mean', np.nan))),
            'GE PCC': float(prnet_recon.get('ge_pred_pearson_mean', np.nan)),
            'DEG-GE PCC': float(prnet_recon.get('DEG_ge_pred_pearson_mean', np.nan)),
            'GE RMSE': float(prnet_recon.get('ge_pred_rmse_mean', np.nan)),
            'n': float(prnet_recon.get('n_samples', np.nan)),
        }
    )
    recon = pd.concat([recon[recon['Model'] != 'PRNet'].copy(), pd.DataFrame([recon_row])], ignore_index=True)

    prnet_downstream = _load_prnet_fixed_downstream_summary()
    prnet_downstream = prnet_downstream[prnet_downstream['model_key'] == 'prnet'].copy()
    prnet_downstream_rows = pd.DataFrame(
        {
            'Subset': prnet_downstream['subset'].astype(str),
            'TopK': prnet_downstream['top_k'].astype(int),
            'Model': 'PRNet',
            'Overlap': prnet_downstream['mean_overlap_ratio'].astype(float),
            'Sign': prnet_downstream['mean_sign_recall'].astype(float),
        }
    )
    downstream = pd.concat([downstream[downstream['Model'] != 'PRNet'].copy(), prnet_downstream_rows], ignore_index=True)

    prnet_fit = _load_prnet_fixed_fit_summary()
    prnet_fit_row = prnet_fit[prnet_fit['model_key'] == 'prnet'].iloc[0]
    fit_row = fit[fit['Model'] == 'PRNet'].iloc[0].to_dict() if (fit['Model'] == 'PRNet').any() else {'Model': 'PRNet'}
    fit_row.update(
        {
            'Model': 'PRNet',
            'All-gene PCC': float(prnet_fit_row['mean_all_gene_pcc']),
            'Top50 overlap': float(prnet_fit_row['mean_top50_overlap_ratio']),
            'Top100 overlap': float(prnet_fit_row['mean_top100_overlap_ratio']),
            'Pathway PCC': float(prnet_fit_row['mean_pathway_gene_pcc']),
        }
    )
    fit = pd.concat([fit[fit['Model'] != 'PRNet'].copy(), pd.DataFrame([fit_row])], ignore_index=True)
    return recon, downstream, fit


def _load_local_analysis_csv(name: str) -> pd.DataFrame:
    path = ANALYSIS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing local Figure 2 analysis table: {path}")
    return pd.read_csv(path).replace(LEGACY_LABEL_MAP)


def _load_plot_inputs_from_local_analysis() -> dict[str, pd.DataFrame]:
    dotplot = _load_local_analysis_csv("figure2_sota_dotplot_values.csv")
    fit = dotplot.rename(columns={"DEG-GE PCC": "All-gene PCC", "Pathway PCC": "Pathway PCC"}).copy()
    if "All-gene PCC" not in fit:
        raise KeyError("figure2_sota_dotplot_values.csv must contain DEG-GE PCC")
    return {
        "fit": fit,
        "recovery_ci": _load_local_analysis_csv("figure2_recovery_error_analysis.csv"),
        "panel_c_seed": _load_local_analysis_csv("figure2_panel_c_multiseed_seed_metrics_used.csv"),
        "bbbc036_deg_seed": _load_local_analysis_csv("figure2_bbbc036_external_deg_seed_metrics.csv"),
    }


def plot_figure2() -> None:
    if USE_LOCAL_ANALYSIS:
        inputs = _load_plot_inputs_from_local_analysis()
        fit = inputs["fit"]
        recovery_ci = inputs["recovery_ci"]
        panel_c_seed = inputs["panel_c_seed"]
        bbbc036_deg_seed = inputs["bbbc036_deg_seed"]
    else:
        recon, downstream, fit, _, sample = load_bbbc047_tables()
        recon = recon[~recon['Model'].isin(FIG1_EXCLUDED_METHODS)].copy()
        downstream = downstream[~downstream['Model'].isin(FIG1_EXCLUDED_METHODS)].copy()
        fit = fit[~fit['Model'].isin(FIG1_EXCLUDED_METHODS)].copy()
        recon, downstream, fit = _replace_prnet_rows(recon, downstream, fit)
        benchmark_metrics = [
            ('GE PCC', 'GE PCC', 'GE reconstruction'),
            ('DEG-GE PCC', 'DEG-GE PCC', 'GE reconstruction'),
            ('Overlap@50', 'Informative overlap@50', 'Downstream'),
            ('Overlap@100', 'Informative overlap@100', 'Downstream'),
            ('Pathway PCC', 'Pathway PCC', 'Pathway-facing'),
        ]
        values = []
        for method in FIG1_GE_METHODS:
            row = {'Model': method}
            for key, _, _ in benchmark_metrics:
                if key in ['GE PCC', 'DEG-GE PCC']:
                    row[key] = recon.loc[recon.Model == method, key].iloc[0]
                elif key.startswith('Overlap@'):
                    topk = int(key.split('@')[1])
                    row[key] = downstream[(downstream.Model == method) & (downstream.TopK == topk) & (downstream.Subset == 'informative')]['Overlap'].iloc[0]
                else:
                    row[key] = fit.loc[fit.Model == method, key].iloc[0]
            values.append(row)
        df = pd.DataFrame(values)
        df.to_csv(ANALYSIS_DIR / 'figure2_sota_dotplot_values.csv', index=False)

        recovery_ci = compute_recovery_error_table()
        panel_c_seed = load_panel_c_seed_metrics()
        panel_c_seed.to_csv(ANALYSIS_DIR / 'figure2_panel_c_multiseed_seed_metrics_used.csv', index=False)

        recovery_ci.to_csv(ANALYSIS_DIR / 'figure2_recovery_error_analysis.csv', index=False)

        bbbc036_deg_seed = load_bbbc036_external_deg_seed_metrics()
        bbbc036_deg_seed.to_csv(ANALYSIS_DIR / 'figure2_bbbc036_external_deg_seed_metrics.csv', index=False)

    fig = plt.figure(figsize=(12.2, 12.1))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0, 1.0], wspace=0.38, hspace=0.62)

    ax = fig.add_subplot(gs[0, 0])
    plot_panel_c_bar_subset(ax, panel_c_seed, endpoint='GE PCC', title='A. BBBC047 GE Reconstruction', min_span=0.14, xlabel='PCC')

    ax = fig.add_subplot(gs[0, 1])
    plot_panel_c_bar_subset(ax, panel_c_seed, endpoint='CP PCC', title='B. BBBC047 CP Reconstruction', min_span=0.18, xlabel='PCC')

    label_styles = {
        'Baseline': {'dx': 0.012, 'dy': -0.010, 'ha': 'left', 'va': 'center'},
        'MVCPert-HyperGate': {'dx': 0.012, 'dy': 0.006, 'ha': 'left', 'va': 'center'},
        'MVCPert': {'dx': 0.016, 'dy': 0.010, 'ha': 'left', 'va': 'center'},
        'PRNet': {'dx': 0.016, 'dy': -0.014, 'ha': 'left', 'va': 'center'},
        'TranSiGen': {'dx': 0.016, 'dy': -0.006, 'ha': 'left', 'va': 'center'},
        'chemCPA': {'dx': 0.015, 'dy': -0.004, 'ha': 'left', 'va': 'center'},
        'cycleCDR': {'dx': -0.012, 'dy': -0.010, 'ha': 'right', 'va': 'center'},
        'MiTCP': {'dx': 0.016, 'dy': 0.008, 'ha': 'left', 'va': 'center'},
        'XPert': {'dx': 0.015, 'dy': 0.010, 'ha': 'left', 'va': 'center'},
    }

    ax = fig.add_subplot(gs[1, 0])
    for method in FIG1_GE_METHODS:
        sub = recovery_ci[(recovery_ci.Model == method) & (recovery_ci.Subset == 'informative')].sort_values('TopK')
        if sub.empty:
            continue
        ax.plot(
            sub['TopK'],
            sub['Overlap'],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            lw=2.0 if method == 'MVCPert' else 1.35,
            ms=7.5 if method == 'MVCPert' else 5.5,
            alpha=0.96,
            label=PANEL_METHOD_LABELS[method],
        )
    ax.set_xticks([20, 50, 100])
    ax.set_xlabel('Top-k Informative Genes', fontsize=12.5)
    ax.set_ylabel('Recovery Overlap', fontsize=12.5)
    ax.set_ylim(0.03, max(0.35, float(recovery_ci['Overlap'].max()) + 0.02))
    ax.set_title('C. Informative Gene Recovery', fontsize=14.2, pad=8)
    ax.grid(True, axis='both')
    ax.tick_params(axis='both', labelsize=11.5)
    clean_axes(ax)

    ax = fig.add_subplot(gs[1, 1])
    for method in FIG1_GE_METHODS:
        if method not in fit.Model.values:
            continue
        row = fit[fit.Model == method].iloc[0]
        ax.scatter(
            row['All-gene PCC'],
            row['Pathway PCC'],
            marker=METHOD_MARKERS[method],
            s=122 if method == 'MVCPert' else 84,
            color=METHOD_COLORS[method],
            edgecolor='white',
            linewidth=0.8,
            zorder=3,
        )
        label_style = label_styles[method]
        ax.text(
            row['All-gene PCC'] + label_style['dx'],
            row['Pathway PCC'] + label_style['dy'],
            PANEL_METHOD_LABELS.get(method, method),
            fontsize=10.4,
            color=METHOD_COLORS[method],
            fontweight='bold',
            ha=label_style['ha'],
            va=label_style['va'],
        )
    # ax.text(0.02, 0.96, '19 informative compounds', transform=ax.transAxes, ha='left', va='top', fontsize=8.8, color='#666666')
    ax.set_xlabel('Informative-subset DEG-PCC', fontsize=12.5)
    ax.set_ylabel('Pathway DEG-PCC', fontsize=12.5)
    ax.set_xlim(-0.02, 0.66)
    ax.set_ylim(-0.02, 0.27)
    ax.set_title('D. Pathway-Level Predictive Fidelity', fontsize=14.2, pad=8)
    ax.grid(True, axis='both')
    ax.tick_params(axis='both', labelsize=11.5)
    clean_axes(ax)

    ax = fig.add_subplot(gs[2, 0])
    plot_panel_c_bar_subset(
        ax,
        bbbc036_deg_seed,
        endpoint='DEG-GE PCC',
        title='E. BBBC036 GE Perturbation Recovery',
        min_span=0.18,
        xlabel='DEG-PCC',
    )

    ax = fig.add_subplot(gs[2, 1])
    plot_panel_c_bar_subset(
        ax,
        bbbc036_deg_seed,
        endpoint='DEG-CP PCC',
        title='F. BBBC036 CP Perturbation Recovery',
        min_span=0.18,
        xlabel='DEG-PCC',
    )

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            lw=2.0 if method == 'MVCPert' else 1.35,
            markersize=8.0 if method == 'MVCPert' else 6.5,
            markeredgecolor='white',
            markeredgewidth=0.5,
            label=PANEL_METHOD_LABELS[method],
        )
        for method in FIG1_GE_METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.365),
        ncol=len(FIG1_GE_METHODS),
        frameon=False,
        fontsize=10.8,
        columnspacing=0.9,
        handlelength=1.4,
        handletextpad=0.4,
    )

    fig.subplots_adjust(top=0.945, bottom=0.07, left=0.09, right=0.985)
    save_figure(fig, 'figure2_multimodal_formal_panels')
    plt.close(fig)


def main() -> None:
    init_plot_env()
    plot_figure2()


if __name__ == '__main__':
    main()
