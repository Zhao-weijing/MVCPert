#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import sys


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathway_benchmark import bbbc047_pathway_benchmark as bm


OUTPUT_DIR = ROOT / "outputs"
MODEL_KEYS_DEFAULT = ["baseline_multi", "dynamics_modulation", "mvcpert"]


ROUND2_DEFAULT_PROFILES = {
    "baseline_multi": Path(
        "<ARTIFACT_ROOT>/2026-04-07/MVC_splitlock_official_v1_round2_baseline_BBBC047_smiles_split/ECFP4_Default/predict/test_prediction_profile.h5"
    ),
    "dynamics_modulation": Path(
        "<ARTIFACT_ROOT>/2026-04-07/MVC_Dynamics_Modulation_splitlock_official_v1_round2_dynamics_BBBC047_smiles_split/ECFP4_Default/predict/test_prediction_profile.h5"
    ),
    "mvcpert": Path(
        "<ARTIFACT_ROOT>/2026-04-23/MVC_HyperGateResidualVAE_main_BBBC047_smiles_split/ECFP4_Default/predict/test_prediction_profile.h5"
    ),
}


def safe_pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def safe_rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_scale_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_norm = float(np.linalg.norm(np.asarray(pred), ord=2))
    gt_norm = float(np.linalg.norm(np.asarray(gt), ord=2))
    if gt_norm <= 1e-12:
        return 0.0
    return pred_norm / gt_norm


def _topk_indices_by_abs(arr: np.ndarray, top_k: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.size == 0 or top_k <= 0:
        return np.array([], dtype=np.int64)
    k = min(int(top_k), values.size)
    idx = np.argpartition(np.abs(values), -k)[-k:]
    order = np.argsort(-np.abs(values[idx]))
    return idx[order]


def compute_topk_overlap_ratio(gt: np.ndarray, pred: np.ndarray, top_k: int) -> float:
    gt_idx = set(_topk_indices_by_abs(gt, top_k).tolist())
    pred_idx = set(_topk_indices_by_abs(pred, top_k).tolist())
    denom = max(len(gt_idx), len(pred_idx))
    if denom == 0:
        return 1.0
    return float(len(gt_idx & pred_idx) / denom)


def compute_topk_sign_recall(gt: np.ndarray, pred: np.ndarray, top_k: int) -> float:
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    gt_idx = _topk_indices_by_abs(gt, top_k)
    if gt_idx.size == 0:
        return 1.0
    gt_sign = np.sign(gt[gt_idx])
    pred_sign = np.sign(pred[gt_idx])
    return float(np.mean(gt_sign == pred_sign))


def compute_topk_abs_mean(arr: np.ndarray, top_k: int) -> float:
    arr = np.asarray(arr, dtype=np.float32)
    idx = _topk_indices_by_abs(arr, top_k)
    if idx.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr[idx])))


def build_pathway_gene_index_map(gene_symbols: Sequence[str], gene_sets: Mapping[str, Set[str]]) -> Dict[str, np.ndarray]:
    symbol_to_index = {str(symbol): idx for idx, symbol in enumerate(np.asarray(gene_symbols).astype(str))}
    pathway_index_map: Dict[str, np.ndarray] = {}
    for term, genes in gene_sets.items():
        indices = [symbol_to_index[g] for g in genes if g in symbol_to_index]
        if indices:
            pathway_index_map[str(term)] = np.array(sorted(set(indices)), dtype=np.int64)
    return pathway_index_map


def build_split_evidence_table(profile_paths: Mapping[str, Path]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_key, profile_path in profile_paths.items():
        run_summary_path = profile_path.parent.parent / "run_summary.json"
        row: Dict[str, object] = {
            "model_key": model_key,
            "profile_path": str(profile_path),
            "run_summary_path": str(run_summary_path),
            "split_source": None,
            "split_lock_path": None,
            "train_unique_smiles": None,
            "valid_unique_smiles": None,
            "test_unique_smiles": None,
        }
        if run_summary_path.exists():
            payload = json.loads(run_summary_path.read_text(encoding="utf-8"))
            dataset_sizes = payload.get("dataset_sizes", {})
            row.update(
                {
                    "split_source": dataset_sizes.get("split_source"),
                    "split_lock_path": dataset_sizes.get("split_lock_path"),
                    "train_unique_smiles": dataset_sizes.get("train_unique_smiles"),
                    "valid_unique_smiles": dataset_sizes.get("valid_unique_smiles"),
                    "test_unique_smiles": dataset_sizes.get("test_unique_smiles"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_diagnosis(
    profile_paths: Mapping[str, Path],
    output_dir: Path,
    model_keys: Sequence[str],
    top_k_grid: Sequence[int],
    ground_truth_profile_key: str = "dynamics_modulation",
    pathway_top_k: int = 80,
    qvalue_cutoff: float = 0.05,
    bootstrap_iter: int = 200,
    random_seed: int = 3407,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    key_type = bm.DEFAULT_MERGED_KEY
    moa_index_df = bm.build_moa_index_table_from_merged(Path(profile_paths[ground_truth_profile_key]), key_type=key_type)
    benchmark_df = bm.select_benchmark_subset(moa_index_df, batch_max_per_moa=bm.BATCH_MAX_PER_MOA)
    benchmark_df, align_stats = bm.restrict_benchmark_to_common_keys(
        benchmark_df=benchmark_df,
        profile_paths=profile_paths,
        key_type=key_type,
        model_keys=model_keys,
    )
    print(f"[diagnose] benchmark common-key stats: {align_stats}")

    gene_symbols = bm.load_gene_symbols()
    knowledge_base = bm.load_default_moa_knowledge_base()
    kegg_sets = bm.load_kegg_gene_sets(bm.KEGG_GMT_PATH)
    pathway_idx_map = build_pathway_gene_index_map(gene_symbols, kegg_sets)

    gt_data = bm.load_subset_data_dict("ground_truth", benchmark_df, profile_paths=profile_paths)
    model_data = {model_key: bm.load_subset_data_dict(model_key, benchmark_df, profile_paths=profile_paths) for model_key in model_keys}

    # Pathway-level metrics (top-k sensitivity primary uses this function)
    sample_pathway_metrics_df, _ = bm.build_sample_pathway_details(
        benchmark_df=benchmark_df,
        gene_symbols=gene_symbols,
        gene_sets=kegg_sets,
        profile_paths=profile_paths,
        model_keys=model_keys,
        knowledge_base=knowledge_base,
        null_frequency_map={},
        top_k=pathway_top_k,
        qvalue_cutoff=qvalue_cutoff,
    )
    ground_truth_rows = sample_pathway_metrics_df.loc[sample_pathway_metrics_df["model_key"] == "ground_truth"].copy()
    informative_ids = set(
        ground_truth_rows.loc[pd.to_numeric(ground_truth_rows["gt_total_count"], errors="coerce").fillna(0) > 0, "sample_id"]
        .astype(int)
        .tolist()
    )
    gt_abs_values_informative = np.abs(
        gt_data["delta_ge"][
            [idx for idx, row in benchmark_df.reset_index(drop=True).iterrows() if int(row["sample_index"]) in informative_ids]
        ]
    ).reshape(-1)
    strong_threshold = float(np.quantile(gt_abs_values_informative, 0.9)) if gt_abs_values_informative.size > 0 else 0.0
    print(f"[diagnose] informative sample count={len(informative_ids)}, strong_threshold(abs delta q90)={strong_threshold:.6f}")

    pathway_metric_lookup = sample_pathway_metrics_df.set_index(["sample_id", "model_key"])
    sample_rows: List[Dict[str, object]] = []
    for local_idx, row in benchmark_df.reset_index(drop=True).iterrows():
        sample_id = int(row["sample_index"])
        moa = str(row["moa"])
        gt_delta = np.asarray(gt_data["delta_ge"][local_idx], dtype=np.float32)
        gt_l2 = float(np.linalg.norm(gt_delta, ord=2))
        gt_l1 = float(np.linalg.norm(gt_delta, ord=1))
        knowledge_terms = bm.get_knowledge_terms_for_moa(moa, knowledge_base=knowledge_base)
        pathway_genes_idx: Set[int] = set()
        for term in knowledge_terms:
            if term in pathway_idx_map:
                pathway_genes_idx.update(pathway_idx_map[term].tolist())
        pathway_idx = np.array(sorted(pathway_genes_idx), dtype=np.int64) if pathway_genes_idx else np.array([], dtype=np.int64)

        for model_key in model_keys:
            pred_delta = np.asarray(model_data[model_key]["delta_ge"][local_idx], dtype=np.float32)
            pred_l2 = float(np.linalg.norm(pred_delta, ord=2))
            pred_l1 = float(np.linalg.norm(pred_delta, ord=1))
            rec = {
                "sample_id": sample_id,
                "smiles": str(row["smiles"]),
                "moa": moa,
                "model_key": model_key,
                "is_informative": int(sample_id in informative_ids),
                "gt_l2_norm": gt_l2,
                "gt_l1_norm": gt_l1,
                "pred_l2_norm": pred_l2,
                "pred_l1_norm": pred_l1,
                "pred_gt_l2_ratio": compute_scale_ratio(pred_delta, gt_delta),
                "pred_top20_abs_mean": compute_topk_abs_mean(pred_delta, 20),
                "pred_top50_abs_mean": compute_topk_abs_mean(pred_delta, 50),
                "pred_top100_abs_mean": compute_topk_abs_mean(pred_delta, 100),
                "gt_top20_abs_mean": compute_topk_abs_mean(gt_delta, 20),
                "gt_top50_abs_mean": compute_topk_abs_mean(gt_delta, 50),
                "gt_top100_abs_mean": compute_topk_abs_mean(gt_delta, 100),
                "pred_strong_gene_count": int(np.sum(np.abs(pred_delta) >= strong_threshold)),
                "gt_strong_gene_count": int(np.sum(np.abs(gt_delta) >= strong_threshold)),
                "all_gene_pcc": safe_pcc(pred_delta, gt_delta),
                "all_gene_rmse": safe_rmse(pred_delta, gt_delta),
                "top20_sign_recall": compute_topk_sign_recall(gt_delta, pred_delta, 20),
                "top50_sign_recall": compute_topk_sign_recall(gt_delta, pred_delta, 50),
                "top100_sign_recall": compute_topk_sign_recall(gt_delta, pred_delta, 100),
                "top50_overlap_ratio": compute_topk_overlap_ratio(gt_delta, pred_delta, 50),
                "top100_overlap_ratio": compute_topk_overlap_ratio(gt_delta, pred_delta, 100),
                "knowledge_term_count": int(len(knowledge_terms)),
                "pathway_gene_count": int(len(pathway_idx)),
                "pathway_gene_pcc": 0.0,
                "pathway_gene_rmse": 0.0,
                "pathway_gene_sign_recall": 0.0,
                "pathway_jaccard": np.nan,
                "pred_total_count": np.nan,
            }
            if pathway_idx.size > 0:
                rec["pathway_gene_pcc"] = safe_pcc(pred_delta[pathway_idx], gt_delta[pathway_idx])
                rec["pathway_gene_rmse"] = safe_rmse(pred_delta[pathway_idx], gt_delta[pathway_idx])
                rec["pathway_gene_sign_recall"] = float(np.mean(np.sign(pred_delta[pathway_idx]) == np.sign(gt_delta[pathway_idx])))
            key = (sample_id, model_key)
            if key in pathway_metric_lookup.index:
                row_metric = pathway_metric_lookup.loc[key]
                rec["pathway_jaccard"] = float(row_metric["pathway_jaccard"])
                rec["pred_total_count"] = float(row_metric["pred_total_count"])
            sample_rows.append(rec)

    sample_df = pd.DataFrame(sample_rows)
    informative_df = sample_df.loc[sample_df["is_informative"] == 1].copy()

    summary_df = (
        informative_df.groupby("model_key", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            mean_pred_gt_l2_ratio=("pred_gt_l2_ratio", "mean"),
            mean_pred_l2_norm=("pred_l2_norm", "mean"),
            mean_gt_l2_norm=("gt_l2_norm", "mean"),
            mean_pred_top20_abs=("pred_top20_abs_mean", "mean"),
            mean_pred_top50_abs=("pred_top50_abs_mean", "mean"),
            mean_pred_top100_abs=("pred_top100_abs_mean", "mean"),
            mean_pred_strong_gene_count=("pred_strong_gene_count", "mean"),
            mean_gt_strong_gene_count=("gt_strong_gene_count", "mean"),
            mean_all_gene_pcc=("all_gene_pcc", "mean"),
            mean_all_gene_rmse=("all_gene_rmse", "mean"),
            mean_top50_sign_recall=("top50_sign_recall", "mean"),
            mean_top100_sign_recall=("top100_sign_recall", "mean"),
            mean_top50_overlap_ratio=("top50_overlap_ratio", "mean"),
            mean_top100_overlap_ratio=("top100_overlap_ratio", "mean"),
            mean_pathway_gene_pcc=("pathway_gene_pcc", "mean"),
            mean_pathway_gene_sign_recall=("pathway_gene_sign_recall", "mean"),
            mean_pathway_jaccard=("pathway_jaccard", "mean"),
            informative_pred_non_empty_count=("pred_total_count", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
        )
        .sort_values("model_key")
        .reset_index(drop=True)
    )

    sensitivity_rows: List[Dict[str, object]] = []
    for top_k in sorted(set(int(v) for v in top_k_grid if int(v) > 0)):
        topk_metrics_df, _ = bm.build_sample_pathway_details(
            benchmark_df=benchmark_df,
            gene_symbols=gene_symbols,
            gene_sets=kegg_sets,
            profile_paths=profile_paths,
            model_keys=model_keys,
            knowledge_base=knowledge_base,
            null_frequency_map={},
            top_k=top_k,
            qvalue_cutoff=qvalue_cutoff,
        )
        topk_info_df = topk_metrics_df.loc[topk_metrics_df["sample_id"].astype(int).isin(informative_ids)].copy()
        for model_key, group_df in topk_info_df.groupby("model_key", sort=False):
            sensitivity_rows.append(
                {
                    "top_k_genes": int(top_k),
                    "model_key": str(model_key),
                    "sample_count": int(group_df["sample_id"].nunique()),
                    "pred_non_empty_count": int((pd.to_numeric(group_df["pred_total_count"], errors="coerce").fillna(0) > 0).sum()),
                    "pred_non_empty_ratio": float(
                        (pd.to_numeric(group_df["pred_total_count"], errors="coerce").fillna(0) > 0).mean()
                    ),
                    "mean_pathway_jaccard": float(pd.to_numeric(group_df["pathway_jaccard"], errors="coerce").mean()),
                    "mean_top_10_overlap": float(pd.to_numeric(group_df["top_10_overlap"], errors="coerce").mean()),
                }
            )
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    rng = np.random.RandomState(random_seed)
    bootstrap_rows: List[Dict[str, object]] = []
    model_metric_df = informative_df.loc[informative_df["model_key"].isin(model_keys), ["sample_id", "model_key", "pathway_jaccard"]].copy()
    sample_ids = sorted(model_metric_df["sample_id"].astype(int).unique().tolist())
    if sample_ids:
        win_counter = {k: 0 for k in model_keys}
        for _ in range(int(bootstrap_iter)):
            sampled = rng.choice(sample_ids, size=len(sample_ids), replace=True)
            sampled_df = (
                pd.DataFrame({"sample_id": sampled})
                .merge(model_metric_df, on="sample_id", how="left")
                .dropna(subset=["model_key"])
            )
            means = sampled_df.groupby("model_key")["pathway_jaccard"].mean().to_dict()
            if means:
                winner = max(means.items(), key=lambda kv: kv[1])[0]
                win_counter[winner] += 1
        for model_key in model_keys:
            bootstrap_rows.append(
                {
                    "model_key": model_key,
                    "bootstrap_iter": int(bootstrap_iter),
                    "win_count_by_mean_pathway_jaccard": int(win_counter[model_key]),
                    "win_rate_by_mean_pathway_jaccard": float(win_counter[model_key] / max(int(bootstrap_iter), 1)),
                }
            )
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    split_evidence_df = build_split_evidence_table(profile_paths)
    split_evidence_df["benchmark_alignment_sample_count_before"] = int(align_stats.get("sample_count_before", 0))
    split_evidence_df["benchmark_alignment_sample_count_after"] = int(align_stats.get("sample_count_after", 0))

    outputs = {
        "model_summary_csv": output_dir / "bbbc047_fit_pathway_gap_model_summary.csv",
        "sample_metrics_csv": output_dir / "bbbc047_fit_pathway_gap_sample_metrics.csv",
        "topk_sensitivity_csv": output_dir / "bbbc047_fit_pathway_gap_topk_sensitivity.csv",
        "bootstrap_csv": output_dir / "bbbc047_fit_pathway_gap_bootstrap_rank_stability.csv",
        "split_evidence_csv": output_dir / "bbbc047_fit_pathway_gap_split_evidence.csv",
    }
    summary_df.to_csv(outputs["model_summary_csv"], index=False)
    sample_df.to_csv(outputs["sample_metrics_csv"], index=False)
    sensitivity_df.to_csv(outputs["topk_sensitivity_csv"], index=False)
    bootstrap_df.to_csv(outputs["bootstrap_csv"], index=False)
    split_evidence_df.to_csv(outputs["split_evidence_csv"], index=False)

    for key, path in outputs.items():
        print(f"[diagnose] {key}: {path}")
    return outputs


def parse_profile_mappings(items: Sequence[str]) -> Dict[str, Path]:
    profile_paths: Dict[str, Path] = {}
    for raw in items:
        if "=" not in str(raw):
            raise ValueError(f"Invalid --profile item (expected model_key=path): {raw}")
        model_key, path_str = str(raw).split("=", 1)
        model_key = model_key.strip()
        if not model_key:
            raise ValueError(f"Empty model_key in --profile item: {raw}")
        profile_paths[model_key] = Path(path_str).expanduser()
    return profile_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose BBBC047 model-fit vs pathway-gap on unified split-lock profiles.")
    parser.add_argument("--baseline-profile", type=Path, default=ROUND2_DEFAULT_PROFILES["baseline_multi"])
    parser.add_argument("--dynamics-profile", type=Path, default=ROUND2_DEFAULT_PROFILES["dynamics_modulation"])
    parser.add_argument("--mvcpert-profile", type=Path, default=ROUND2_DEFAULT_PROFILES["mvcpert"])
    parser.add_argument("--profile", action="append", default=[], help="Custom profile mapping: model_key=/path/to/test_prediction_profile.h5")
    parser.add_argument("--model-keys", type=str, default="", help="Comma-separated model keys; defaults to builtin order or keys from --profile")
    parser.add_argument("--ground-truth-profile-key", type=str, default="dynamics_modulation")
    parser.add_argument("--top-k-grid", type=str, default="40,80,120,200")
    parser.add_argument("--pathway-top-k", type=int, default=80)
    parser.add_argument("--qvalue-cutoff", type=float, default=0.05)
    parser.add_argument("--bootstrap-iter", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=3407)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> Dict[str, Path]:
    args = parse_args()
    top_k_grid = [int(v.strip()) for v in str(args.top_k_grid).split(",") if v.strip()]
    profile_paths: Dict[str, Path] = {
        "baseline_multi": Path(args.baseline_profile),
        "dynamics_modulation": Path(args.dynamics_profile),
        "mvcpert": Path(args.mvcpert_profile),
    }
    if args.profile:
        profile_paths = parse_profile_mappings(args.profile)
    if str(args.model_keys).strip():
        model_keys = [v.strip() for v in str(args.model_keys).split(",") if v.strip()]
    else:
        model_keys = list(profile_paths.keys()) if args.profile else list(MODEL_KEYS_DEFAULT)
    return run_diagnosis(
        profile_paths=profile_paths,
        output_dir=Path(args.output_dir),
        model_keys=model_keys,
        top_k_grid=top_k_grid,
        ground_truth_profile_key=str(args.ground_truth_profile_key),
        pathway_top_k=int(args.pathway_top_k),
        qvalue_cutoff=float(args.qvalue_cutoff),
        bootstrap_iter=int(args.bootstrap_iter),
        random_seed=int(args.random_seed),
    )


if __name__ == "__main__":
    main()
