#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathway_benchmark import bbbc047_pathway_benchmark as bm


OUTPUT_DIR = ROOT / "outputs"
MODEL_KEYS_DEFAULT = ["baseline_multi", "dynamics_modulation", "mvcpert"]
TOP_K_LIST_DEFAULT = [20, 50, 100]

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


def build_pairwise_advantage_table(
    summary_df: pd.DataFrame,
    reference_model: str,
    comparison_models: Sequence[str],
) -> pd.DataFrame:
    info_df = summary_df.loc[summary_df["subset"] == "informative"].copy()
    rows: List[Dict[str, float]] = []
    for top_k, group_df in info_df.groupby("top_k", sort=True):
        value_map = group_df.set_index("model_key")
        if reference_model not in value_map.index:
            continue
        for model_key in comparison_models:
            if model_key not in value_map.index:
                continue
            rows.append(
                {
                    "top_k": int(top_k),
                    "reference_model": str(reference_model),
                    "comparison_model": str(model_key),
                    "overlap_advantage": float(
                        value_map.loc[model_key, "mean_overlap_ratio"] - value_map.loc[reference_model, "mean_overlap_ratio"]
                    ),
                    "sign_recall_advantage": float(
                        value_map.loc[model_key, "mean_sign_recall"] - value_map.loc[reference_model, "mean_sign_recall"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("top_k").reset_index(drop=True) if rows else pd.DataFrame()


def build_bootstrap_winrate_table(
    sample_df: pd.DataFrame,
    model_keys: Sequence[str],
    reference_model: str,
    top_k_list: Sequence[int],
    bootstrap_iter: int = 500,
    random_seed: int = 3407,
) -> pd.DataFrame:
    rng = np.random.RandomState(random_seed)
    info_df = sample_df.loc[sample_df["subset"] == "informative"].copy()
    rows: List[Dict[str, float]] = []
    comparison_models = [m for m in model_keys if m != reference_model]
    for top_k in top_k_list:
        k_df = info_df.loc[info_df["top_k"] == int(top_k)].copy()
        sample_ids = sorted(k_df["sample_id"].astype(int).unique().tolist())
        if not sample_ids:
            continue
        base_df = k_df.loc[k_df["model_key"] == reference_model, ["sample_id", "overlap_ratio", "sign_recall"]].rename(
            columns={"overlap_ratio": "base_overlap", "sign_recall": "base_sign"}
        )
        for comp in comparison_models:
            comp_df = k_df.loc[k_df["model_key"] == comp, ["sample_id", "overlap_ratio", "sign_recall"]].rename(
                columns={"overlap_ratio": "comp_overlap", "sign_recall": "comp_sign"}
            )
            merged = base_df.merge(comp_df, on="sample_id", how="inner")
            if merged.empty:
                continue
            overlap_win = 0
            sign_win = 0
            for _ in range(int(bootstrap_iter)):
                sampled_ids = rng.choice(merged["sample_id"].to_numpy(), size=len(merged), replace=True)
                sampled = pd.DataFrame({"sample_id": sampled_ids}).merge(merged, on="sample_id", how="left")
                if sampled["comp_overlap"].mean() > sampled["base_overlap"].mean():
                    overlap_win += 1
                if sampled["comp_sign"].mean() > sampled["base_sign"].mean():
                    sign_win += 1
            rows.append(
                {
                    "top_k": int(top_k),
                    "reference_model": str(reference_model),
                    "comparison_model": str(comp),
                    "comparison": f"{comp}_vs_{reference_model}",
                    "bootstrap_iter": int(bootstrap_iter),
                    "overlap_win_rate": float(overlap_win / max(int(bootstrap_iter), 1)),
                    "sign_recall_win_rate": float(sign_win / max(int(bootstrap_iter), 1)),
                }
            )
    return pd.DataFrame(rows)


def run_evaluation(
    profile_paths: Mapping[str, Path],
    output_dir: Path,
    model_keys: Sequence[str],
    top_k_list: Sequence[int],
    ground_truth_profile_key: str = "dynamics_modulation",
    reference_model: str = "baseline_multi",
    informative_top_k: int = 80,
    qvalue_cutoff: float = 0.05,
    bootstrap_iter: int = 500,
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
    print(f"[gene-level] benchmark alignment: {align_stats}")

    gene_symbols = bm.load_gene_symbols()
    gene_sets = bm.load_kegg_gene_sets(bm.KEGG_GMT_PATH)
    knowledge_base = bm.load_default_moa_knowledge_base()
    sample_pathway_metrics_df, _ = bm.build_sample_pathway_details(
        benchmark_df=benchmark_df,
        gene_symbols=gene_symbols,
        gene_sets=gene_sets,
        profile_paths=profile_paths,
        model_keys=model_keys,
        knowledge_base=knowledge_base,
        null_frequency_map={},
        top_k=int(informative_top_k),
        qvalue_cutoff=float(qvalue_cutoff),
    )
    gt_rows = sample_pathway_metrics_df.loc[sample_pathway_metrics_df["model_key"] == "ground_truth"].copy()
    informative_ids = set(
        gt_rows.loc[pd.to_numeric(gt_rows["gt_total_count"], errors="coerce").fillna(0) > 0, "sample_id"].astype(int).tolist()
    )
    print(f"[gene-level] informative sample count={len(informative_ids)}")

    gt_data = bm.load_subset_data_dict("ground_truth", benchmark_df, profile_paths=profile_paths)
    model_data = {m: bm.load_subset_data_dict(m, benchmark_df, profile_paths=profile_paths) for m in model_keys}

    rows: List[Dict[str, object]] = []
    for local_idx, row in benchmark_df.reset_index(drop=True).iterrows():
        sample_id = int(row["sample_index"])
        gt_delta = np.asarray(gt_data["delta_ge"][local_idx], dtype=np.float32)
        for model_key in model_keys:
            pred_delta = np.asarray(model_data[model_key]["delta_ge"][local_idx], dtype=np.float32)
            for top_k in top_k_list:
                record = {
                    "sample_id": sample_id,
                    "smiles": str(row["smiles"]),
                    "moa": str(row["moa"]),
                    "model_key": model_key,
                    "top_k": int(top_k),
                    "overlap_ratio": compute_topk_overlap_ratio(gt_delta, pred_delta, int(top_k)),
                    "sign_recall": compute_topk_sign_recall(gt_delta, pred_delta, int(top_k)),
                }
                rows.append({"subset": "all", **record})
                if sample_id in informative_ids:
                    rows.append({"subset": "informative", **record})

    sample_df = pd.DataFrame(rows)
    summary_df = (
        sample_df.groupby(["subset", "top_k", "model_key"], as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            mean_overlap_ratio=("overlap_ratio", "mean"),
            mean_sign_recall=("sign_recall", "mean"),
        )
        .sort_values(["subset", "top_k", "model_key"])
        .reset_index(drop=True)
    )
    pairwise_df = build_pairwise_advantage_table(
        summary_df,
        reference_model=reference_model,
        comparison_models=[m for m in model_keys if m != reference_model],
    )
    bootstrap_df = build_bootstrap_winrate_table(
        sample_df,
        model_keys=model_keys,
        reference_model=reference_model,
        top_k_list=top_k_list,
        bootstrap_iter=int(bootstrap_iter),
        random_seed=int(random_seed),
    )

    outputs = {
        "summary_csv": output_dir / "bbbc047_gene_level_downstream_summary.csv",
        "sample_csv": output_dir / "bbbc047_gene_level_downstream_sample_metrics.csv",
        "pairwise_csv": output_dir / "bbbc047_gene_level_downstream_pairwise_advantage.csv",
        "bootstrap_csv": output_dir / "bbbc047_gene_level_downstream_bootstrap_winrate.csv",
    }
    summary_df.to_csv(outputs["summary_csv"], index=False)
    sample_df.to_csv(outputs["sample_csv"], index=False)
    pairwise_df.to_csv(outputs["pairwise_csv"], index=False)
    bootstrap_df.to_csv(outputs["bootstrap_csv"], index=False)
    for k, p in outputs.items():
        print(f"[gene-level] {k}: {p}")
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
    parser = argparse.ArgumentParser(description="Evaluate BBBC047 gene-level downstream metrics on unified split-lock profiles.")
    parser.add_argument("--baseline-profile", type=Path, default=ROUND2_DEFAULT_PROFILES["baseline_multi"])
    parser.add_argument("--dynamics-profile", type=Path, default=ROUND2_DEFAULT_PROFILES["dynamics_modulation"])
    parser.add_argument("--mvcpert-profile", type=Path, default=ROUND2_DEFAULT_PROFILES["mvcpert"])
    parser.add_argument("--profile", action="append", default=[], help="Custom profile mapping: model_key=/path/to/test_prediction_profile.h5")
    parser.add_argument("--model-keys", type=str, default="", help="Comma-separated model keys; defaults to builtin order or keys from --profile")
    parser.add_argument("--ground-truth-profile-key", type=str, default="dynamics_modulation")
    parser.add_argument("--reference-model", type=str, default="baseline_multi")
    parser.add_argument("--top-k-list", type=str, default="20,50,100")
    parser.add_argument("--informative-top-k", type=int, default=80)
    parser.add_argument("--qvalue-cutoff", type=float, default=0.05)
    parser.add_argument("--bootstrap-iter", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=3407)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> Dict[str, Path]:
    args = parse_args()
    top_k_list = [int(v.strip()) for v in str(args.top_k_list).split(",") if v.strip()]
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
    return run_evaluation(
        profile_paths=profile_paths,
        output_dir=Path(args.output_dir),
        model_keys=model_keys,
        top_k_list=top_k_list,
        ground_truth_profile_key=str(args.ground_truth_profile_key),
        reference_model=str(args.reference_model),
        informative_top_k=int(args.informative_top_k),
        qvalue_cutoff=float(args.qvalue_cutoff),
        bootstrap_iter=int(args.bootstrap_iter),
        random_seed=int(args.random_seed),
    )


if __name__ == "__main__":
    main()
