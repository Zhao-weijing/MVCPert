#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import inchi
from scipy.stats import hypergeom

RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

MODEL_LABELS = {
    "baseline_multi": "Baseline Multi",
    "dynamics_modulation": "Dynamics_Modulation",
    "mvcpert": "MVCPert",
    "ground_truth": "Ground Truth",
}
MODEL_ORDER = ["baseline_multi", "dynamics_modulation", "mvcpert", "ground_truth"]

DEFAULT_PROFILE_PATHS = {
    "baseline_multi": Path(
        "<ARTIFACT_ROOT>/2026-04-01/MVC_BBBC047_smiles_split/ECFP4_Default/predict/test_prediction_profile.h5"
    ),
    "dynamics_modulation": Path(
        "<ARTIFACT_ROOT>/2026-04-01/MVC_Dynamics_Modulation_BBBC047_smiles_split/ECFP4_Default/predict/test_prediction_profile.h5"
    ),
    "mvcpert": Path(
        "<ARTIFACT_ROOT>/2026-04-23/MVC_HyperGateResidualVAE_main_BBBC047_smiles_split/ECFP4_Default/predict/test_prediction_profile.h5"
    ),
}

GENE_MAPPING_PATH = Path("<DATA_ROOT>/MVC_BBBC047/gene_map.csv")
DRUG_META_PATH = Path("<DATA_ROOT>/MVC_BBBC047/merged_repurposing.csv")
KEGG_GMT_PATH = Path("<DATA_ROOT>/MVC_BBBC047/enrichr_libraries/KEGG_2021_Human.gmt")
REACTOME_GMT_PATH = Path("<DATA_ROOT>/MVC_BBBC047/enrichr_libraries/Reactome_2022.gmt")
KNOWLEDGE_BASE_PATH = ROOT / "knowledge" / "bbbc047_moa_expected_pathways.csv"

TOP_K_GENES = 80
QVALUE_CUTOFF = 0.05
BATCH_MAX_PER_MOA = 6
BENCHMARK_RANDOM_SEED = 42
DEFAULT_MERGED_KEY = "canon_iso"
MOA_TOKEN_SPLIT_RE = re.compile(r"[|;]")


def decode_array(arr: Sequence[object]) -> np.ndarray:
    return np.array(
        [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
    )


def to_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def smiles_key(smiles: str, key_type: str) -> str | None:
    mol = to_mol(smiles)
    if mol is None:
        return None
    if key_type == "canon_iso":
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if key_type == "canon_noiso":
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    if key_type == "inchikey14":
        ik = inchi.MolToInchiKey(mol)
        return ik.split("-")[0] if ik else None
    raise ValueError(f"Unsupported smiles key type: {key_type}")


def build_profile_subset_alignment(profile_keys: Sequence[object], benchmark_keys: Sequence[object]) -> np.ndarray:
    buckets: MutableMapping[object, deque[int]] = defaultdict(deque)
    for idx, key in enumerate(profile_keys):
        buckets[key].append(idx)

    alignment = np.empty(len(benchmark_keys), dtype=np.int64)
    for idx, key in enumerate(benchmark_keys):
        if not buckets[key]:
            raise ValueError(f"Missing key during profile alignment: {key}")
        alignment[idx] = buckets[key].popleft()
    return alignment


def load_gene_symbols(gene_mapping_path: Path = GENE_MAPPING_PATH) -> np.ndarray:
    gene_df = pd.read_csv(gene_mapping_path)
    if "gene_symbol" not in gene_df.columns:
        raise ValueError(f"gene_map 缺少 gene_symbol 列: {gene_mapping_path}")
    return gene_df["gene_symbol"].astype(str).to_numpy()


def load_kegg_gene_sets(gmt_path: Path = KEGG_GMT_PATH) -> Dict[str, Set[str]]:
    gene_sets: Dict[str, Set[str]] = {}
    with open(gmt_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = {gene.strip() for gene in parts[2:] if gene.strip()}
            if genes:
                gene_sets[term] = genes
    if not gene_sets:
        raise ValueError(f"No KEGG gene sets loaded from {gmt_path}")
    return gene_sets


def get_default_gene_set_configs() -> Dict[str, Path]:
    return {
        "kegg": KEGG_GMT_PATH,
        "reactome": REACTOME_GMT_PATH,
    }


def normalize_moa_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def parse_moa_tokens(value: object) -> Tuple[str, ...]:
    text = str(value or "")
    tokens: List[str] = []
    for raw_token in MOA_TOKEN_SPLIT_RE.split(text):
        token = normalize_moa_text(raw_token)
        if token:
            tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def load_moa_knowledge_base(knowledge_paths: Sequence[Path] | None = None) -> Dict[str, Set[str]]:
    knowledge_paths = list(knowledge_paths or [KNOWLEDGE_BASE_PATH])
    frames: List[pd.DataFrame] = []
    for knowledge_path in knowledge_paths:
        knowledge_df = pd.read_csv(knowledge_path)
        required_cols = {"moa_keyword", "pathway_name"}
        if not required_cols.issubset(set(knowledge_df.columns)):
            raise ValueError(f"Knowledge base missing required columns: {knowledge_path}")
        frames.append(knowledge_df.loc[:, [col for col in knowledge_df.columns if col in required_cols]].copy())

    if not frames:
        return {}

    knowledge_df = pd.concat(frames, ignore_index=True)
    required_cols = {"moa_keyword", "pathway_name"}
    if not required_cols.issubset(set(knowledge_df.columns)):
        raise ValueError(f"Knowledge base missing required columns: {knowledge_paths}")
    grouped = knowledge_df.dropna(subset=["moa_keyword", "pathway_name"]).groupby("moa_keyword")["pathway_name"]
    return {str(key): set(map(str, values.tolist())) for key, values in grouped}


def load_default_moa_knowledge_base(knowledge_path: Path = KNOWLEDGE_BASE_PATH) -> Dict[str, Set[str]]:
    return load_moa_knowledge_base([knowledge_path])


def get_knowledge_terms_for_moa(moa: str, knowledge_base: Mapping[str, Set[str]] | None = None) -> Set[str]:
    knowledge_base = knowledge_base or load_default_moa_knowledge_base()
    moa_text = normalize_moa_text(moa)
    moa_tokens = set(parse_moa_tokens(moa))
    matched_terms: Set[str] = set()
    for key, terms in knowledge_base.items():
        normalized_key = normalize_moa_text(key)
        if normalized_key and (normalized_key in moa_tokens or normalized_key in moa_text):
            matched_terms.update(terms)
    return matched_terms


def build_moa_index_table_from_merged(
    profile_path: Path,
    drug_meta_path: Path = DRUG_META_PATH,
    key_type: str = DEFAULT_MERGED_KEY,
) -> pd.DataFrame:
    with h5py.File(profile_path, "r") as handle:
        profile_smiles = decode_array(handle["smiles"][:])

    full_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(profile_smiles), dtype=int),
            "smiles": profile_smiles,
        }
    )
    drug_meta_df = pd.read_csv(drug_meta_path, usecols=["smiles", "moa"]).dropna(subset=["smiles", "moa"]).copy()
    full_df["match_key"] = full_df["smiles"].astype(str).map(lambda s: smiles_key(s, key_type))
    drug_meta_df["match_key"] = drug_meta_df["smiles"].astype(str).map(lambda s: smiles_key(s, key_type))
    drug_meta_df = drug_meta_df.dropna(subset=["match_key"]).copy()

    moa_map = (
        drug_meta_df.groupby("match_key")["moa"]
        .agg(lambda series: series.value_counts().index[0])
        .reset_index()
        .rename(columns={"moa": "moa"})
    )
    mapping = full_df.merge(moa_map, on="match_key", how="left").dropna(subset=["moa"]).copy()
    if mapping.empty:
        raise ValueError("No MoA labels could be mapped from merged_repurposing.csv.")
    mapping["key_type"] = key_type
    return mapping[["sample_index", "smiles", "match_key", "moa", "key_type"]].copy()


def select_benchmark_subset(
    moa_index_df: pd.DataFrame,
    batch_max_per_moa: int = BATCH_MAX_PER_MOA,
    random_seed: int = BENCHMARK_RANDOM_SEED,
) -> pd.DataFrame:
    sample_df = moa_index_df.copy()
    sample_df = sample_df.drop_duplicates(["moa", "match_key"], keep="first").copy()

    rng = np.random.RandomState(random_seed)
    selected_indices: List[int] = []
    for _, group_df in sample_df.groupby("moa", sort=False):
        idx_arr = group_df["sample_index"].to_numpy(dtype=int)
        if len(idx_arr) > batch_max_per_moa:
            idx_arr = rng.choice(idx_arr, size=batch_max_per_moa, replace=False)
        selected_indices.extend(idx_arr.tolist())

    selected_df = moa_index_df.loc[moa_index_df["sample_index"].isin(selected_indices)].copy()
    selected_df = selected_df.drop_duplicates("sample_index").copy()
    selected_df = selected_df.set_index("sample_index").loc[selected_indices].reset_index()
    return selected_df.reset_index(drop=True)


def load_profile_arrays(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "control_ge": handle["control_ge"][:].astype(np.float32, copy=False),
            "target_ge": handle["target_ge"][:].astype(np.float32, copy=False),
            "ge_pred": handle["ge_pred"][:].astype(np.float32, copy=False),
            "smiles": decode_array(handle["smiles"][:]).astype(str),
        }


def load_profile_key_set(path: Path, key_type: str) -> Set[object]:
    with h5py.File(path, "r") as handle:
        smiles = decode_array(handle["smiles"][:]).astype(str)
    keys = [smiles_key(s, key_type) for s in smiles]
    return {key for key in keys if key is not None}


def restrict_benchmark_to_common_keys(
    benchmark_df: pd.DataFrame,
    profile_paths: Mapping[str, Path],
    key_type: str,
    model_keys: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    model_keys = list(model_keys or [key for key in MODEL_ORDER if key != "ground_truth"])
    if benchmark_df.empty:
        return benchmark_df.copy(), {"sample_count_before": 0, "sample_count_after": 0}

    key_sets: Dict[str, Set[object]] = {}
    common_keys: Set[object] | None = None
    for model_key in model_keys:
        profile_path = Path(profile_paths[model_key])
        key_set = load_profile_key_set(profile_path, key_type=key_type)
        key_sets[model_key] = key_set
        common_keys = key_set if common_keys is None else (common_keys & key_set)

    common_keys = common_keys or set()
    filtered_df = benchmark_df.loc[benchmark_df["match_key"].isin(common_keys)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    stats: Dict[str, int] = {
        "sample_count_before": int(len(benchmark_df)),
        "sample_count_after": int(len(filtered_df)),
        "common_key_count": int(len(common_keys)),
    }
    for model_key in model_keys:
        model_key_set = key_sets.get(model_key, set())
        missing_count = int((~benchmark_df["match_key"].isin(model_key_set)).sum())
        stats[f"missing_in_{model_key}"] = missing_count
    return filtered_df, stats


def load_subset_data_dict(
    model_key: str,
    benchmark_df: pd.DataFrame,
    profile_paths: Mapping[str, Path] | None = None,
) -> Dict[str, np.ndarray]:
    if profile_paths is None:
        profile_paths = DEFAULT_PROFILE_PATHS
    benchmark_keys = benchmark_df["match_key"].to_numpy(dtype=object)
    key_type = str(benchmark_df["key_type"].iloc[0])

    if model_key == "ground_truth":
        ground_truth_source_key = "dynamics_modulation" if "dynamics_modulation" in profile_paths else next(iter(profile_paths.keys()))
        source_path = Path(profile_paths[ground_truth_source_key])
        arrays = load_profile_arrays(source_path)
        profile_keys = np.array([smiles_key(s, key_type) for s in arrays["smiles"]], dtype=object)
        aligned_indices = build_profile_subset_alignment(profile_keys, benchmark_keys)
        control_ge = arrays["control_ge"][aligned_indices]
        target_ge = arrays["target_ge"][aligned_indices]
        delta_ge = target_ge - control_ge
        smiles = arrays["smiles"][aligned_indices]
    else:
        source_path = Path(profile_paths[model_key])
        arrays = load_profile_arrays(source_path)
        profile_keys = np.array([smiles_key(s, key_type) for s in arrays["smiles"]], dtype=object)
        aligned_indices = build_profile_subset_alignment(profile_keys, benchmark_keys)
        control_ge = arrays["control_ge"][aligned_indices]
        ge_pred = arrays["ge_pred"][aligned_indices]
        delta_ge = ge_pred - control_ge
        smiles = arrays["smiles"][aligned_indices]

    return {
        "delta_ge": np.asarray(delta_ge, dtype=np.float32),
        "smiles": np.asarray(smiles),
        "moa": benchmark_df["moa"].astype(str).to_numpy(),
    }


def get_model_label(model_key: str) -> str:
    return MODEL_LABELS.get(str(model_key), str(model_key))


def get_top_up_down(signature: np.ndarray, gene_symbols: np.ndarray, top_k: int = TOP_K_GENES) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx_sorted = np.argsort(signature)
    down_idx = idx_sorted[:top_k]
    up_idx = idx_sorted[-top_k:][::-1]

    down_df = pd.DataFrame({"gene": gene_symbols[down_idx], "score": signature[down_idx]})
    up_df = pd.DataFrame({"gene": gene_symbols[up_idx], "score": signature[up_idx]})
    return up_df, down_df


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(p_values, dtype=float)
    if pvals.size == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty_like(ranked)
    n = ranked.size
    running = 1.0
    for rev_idx in range(n - 1, -1, -1):
        rank = rev_idx + 1
        value = ranked[rev_idx] * n / rank
        running = min(running, value)
        adjusted[rev_idx] = running
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def compute_odds_ratio(overlap: int, query_size: int, set_size: int, background_size: int) -> float:
    a = float(overlap)
    b = float(max(query_size - overlap, 0))
    c = float(max(set_size - overlap, 0))
    d = float(max(background_size - set_size - query_size + overlap, 0))
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def run_ora(gene_list: Sequence[str], background_genes: Sequence[str], gene_sets: Mapping[str, Set[str]]) -> pd.DataFrame:
    gene_set = set(str(gene) for gene in gene_list if str(gene))
    background = set(str(gene) for gene in background_genes if str(gene))
    if len(gene_set) < 3 or not background:
        return pd.DataFrame(columns=["Term", "P-value", "Adjusted P-value", "Odds Ratio", "Overlap"])

    rows: List[Dict[str, object]] = []
    population_size = len(background)
    query_size = len(gene_set & background)
    for term, term_genes in gene_sets.items():
        overlap_genes = gene_set & term_genes & background
        overlap = len(overlap_genes)
        if overlap == 0:
            continue
        set_size = len(term_genes & background)
        if set_size == 0:
            continue
        p_value = float(hypergeom.sf(overlap - 1, population_size, set_size, query_size))
        rows.append(
            {
                "Term": term,
                "P-value": p_value,
                "Odds Ratio": compute_odds_ratio(overlap, query_size, set_size, population_size),
                "Overlap": f"{overlap}/{set_size}",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Term", "P-value", "Adjusted P-value", "Odds Ratio", "Overlap"])

    result = pd.DataFrame(rows)
    result["Adjusted P-value"] = bh_adjust(result["P-value"].to_numpy())
    return result.sort_values(["Adjusted P-value", "P-value", "Odds Ratio"], ascending=[True, True, False]).reset_index(drop=True)


def enrich_signature(
    signature: np.ndarray,
    gene_symbols: np.ndarray,
    gene_sets: Mapping[str, Set[str]],
    top_k: int = TOP_K_GENES,
    qvalue_cutoff: float = QVALUE_CUTOFF,
) -> Dict[str, pd.DataFrame]:
    up_df, down_df = get_top_up_down(signature, gene_symbols, top_k=top_k)
    background = gene_symbols.tolist()
    up_enr = run_ora(up_df["gene"].tolist(), background, gene_sets)
    down_enr = run_ora(down_df["gene"].tolist(), background, gene_sets)
    up_enr = up_enr.loc[up_enr["Adjusted P-value"] <= qvalue_cutoff].reset_index(drop=True)
    down_enr = down_enr.loc[down_enr["Adjusted P-value"] <= qvalue_cutoff].reset_index(drop=True)
    return {"up": up_enr, "down": down_enr}


def extract_ranked_pathway_terms(enrichment: Mapping[str, pd.DataFrame]) -> List[str]:
    ranked_frames: List[pd.DataFrame] = []
    for direction in ("up", "down"):
        enr_df = enrichment.get(direction, pd.DataFrame())
        if enr_df is None or enr_df.empty or "Term" not in enr_df.columns:
            continue
        work_df = enr_df.copy()
        work_df["direction"] = direction
        ranked_frames.append(work_df)
    if not ranked_frames:
        return []

    merged_df = pd.concat(ranked_frames, ignore_index=True)
    sort_cols = [col for col in ["Adjusted P-value", "Odds Ratio", "Term"] if col in merged_df.columns]
    ascending = [True if col == "Adjusted P-value" else False if col == "Odds Ratio" else True for col in sort_cols]
    merged_df = merged_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ordered_terms: List[str] = []
    seen: Set[str] = set()
    for term in merged_df["Term"].astype(str).tolist():
        if term in seen:
            continue
        seen.add(term)
        ordered_terms.append(term)
    return ordered_terms


def compute_topk_overlap(
    gt_ranked_terms: Sequence[str] | None,
    pred_ranked_terms: Sequence[str] | None,
    k: int,
) -> float:
    gt_top = list(gt_ranked_terms or [])[: max(int(k), 0)]
    pred_top = list(pred_ranked_terms or [])[: max(int(k), 0)]
    denom = max(len(gt_top), len(pred_top))
    if denom == 0:
        return 1.0
    return len(set(gt_top) & set(pred_top)) / denom


def compute_null_adjusted_unsupported_extra_rate(
    unsupported_extra_terms: Set[str],
    pred_total_count: int,
    null_frequency_map: Mapping[str, float] | None = None,
) -> float:
    if pred_total_count <= 0 or not unsupported_extra_terms:
        return 0.0
    null_frequency_map = null_frequency_map or {}
    weighted_penalty = 0.0
    for term in unsupported_extra_terms:
        null_freq = float(null_frequency_map.get(str(term), 0.0))
        null_freq = min(max(null_freq, 0.0), 1.0)
        weighted_penalty += 1.0 - null_freq
    return weighted_penalty / pred_total_count


def compare_sample_pathways(
    gt_up: Set[str],
    gt_down: Set[str],
    pred_up: Set[str],
    pred_down: Set[str],
    knowledge_terms: Set[str] | None = None,
    null_frequency_map: Mapping[str, float] | None = None,
    gt_ranked_terms: Sequence[str] | None = None,
    pred_ranked_terms: Sequence[str] | None = None,
) -> Dict[str, float]:
    knowledge_terms = knowledge_terms or set()
    null_frequency_map = null_frequency_map or {}
    gt_all = set(gt_up) | set(gt_down)
    pred_all = set(pred_up) | set(pred_down)
    overlap_all = gt_all & pred_all
    union_all = gt_all | pred_all

    direction_match_terms = (set(gt_up) & set(pred_up)) | (set(gt_down) & set(pred_down))
    pred_only_terms = pred_all - gt_all
    unsupported_extra_terms = pred_only_terms - set(knowledge_terms)

    pathway_jaccard = 1.0 if not union_all else len(overlap_all) / len(union_all)
    direction_consistency = 1.0 if not pred_all else len(direction_match_terms) / len(pred_all)
    unsupported_extra_rate = 0.0 if not pred_all else len(unsupported_extra_terms) / len(pred_all)
    knowledge_supported_extra_terms = pred_only_terms & set(knowledge_terms)
    knowledge_term_count = len(set(knowledge_terms))
    knowledge_recall = 0.0 if knowledge_term_count == 0 else len(pred_all & set(knowledge_terms)) / knowledge_term_count
    null_adjusted_unsupported_extra_rate = compute_null_adjusted_unsupported_extra_rate(
        unsupported_extra_terms=unsupported_extra_terms,
        pred_total_count=len(pred_all),
        null_frequency_map=null_frequency_map,
    )

    return {
        "gt_total_count": len(gt_all),
        "pred_total_count": len(pred_all),
        "overlap_count": len(overlap_all),
        "pathway_jaccard": pathway_jaccard,
        "direction_match_count": len(direction_match_terms),
        "direction_consistency": direction_consistency,
        "knowledge_term_count": knowledge_term_count,
        "knowledge_supported_extra_count": len(knowledge_supported_extra_terms),
        "knowledge_recall": knowledge_recall,
        "unsupported_extra_count": len(unsupported_extra_terms),
        "unsupported_extra_rate": unsupported_extra_rate,
        "null_adjusted_unsupported_extra_rate": null_adjusted_unsupported_extra_rate,
        "top_5_overlap": compute_topk_overlap(gt_ranked_terms, pred_ranked_terms, 5),
        "top_10_overlap": compute_topk_overlap(gt_ranked_terms, pred_ranked_terms, 10),
        "top_20_overlap": compute_topk_overlap(gt_ranked_terms, pred_ranked_terms, 20),
    }


def summarize_model_metrics(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if sample_metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "model_key",
                "sample_count",
                "mean_pathway_jaccard",
                "mean_direction_consistency",
                "mean_unsupported_extra_rate",
                "mean_null_adjusted_unsupported_extra_rate",
                "mean_top_5_overlap",
                "mean_top_10_overlap",
                "mean_top_20_overlap",
                "mean_knowledge_recall",
            ]
        )

    work_df = sample_metrics_df.copy()
    for metric_col in [
        "null_adjusted_unsupported_extra_rate",
        "top_5_overlap",
        "top_10_overlap",
        "top_20_overlap",
    ]:
        if metric_col not in work_df.columns:
            work_df[metric_col] = 0.0

    return (
        work_df.groupby("model_key", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique") if "sample_id" in work_df.columns else ("model_key", "size"),
            mean_pathway_jaccard=("pathway_jaccard", "mean"),
            mean_direction_consistency=("direction_consistency", "mean"),
            mean_unsupported_extra_rate=("unsupported_extra_rate", "mean"),
            mean_null_adjusted_unsupported_extra_rate=("null_adjusted_unsupported_extra_rate", "mean"),
            mean_top_5_overlap=("top_5_overlap", "mean"),
            mean_top_10_overlap=("top_10_overlap", "mean"),
            mean_top_20_overlap=("top_20_overlap", "mean"),
            mean_knowledge_recall=("knowledge_recall", "mean"),
            mean_direction_match_count=("direction_match_count", "mean"),
            mean_unsupported_extra_count=("unsupported_extra_count", "mean"),
        )
        .sort_values("model_key")
        .reset_index(drop=True)
    )


def filter_informative_subset(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if sample_metrics_df.empty:
        return sample_metrics_df.copy()
    work_df = sample_metrics_df.copy()
    gt_total = pd.to_numeric(work_df.get("gt_total_count", 0), errors="coerce").fillna(0.0)
    return work_df.loc[gt_total > 0].reset_index(drop=True)


def summarize_model_metrics_informative(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    return summarize_model_metrics(filter_informative_subset(sample_metrics_df))


def summarize_moa_metrics_informative(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    return summarize_moa_metrics(filter_informative_subset(sample_metrics_df))


def summarize_signal_audit(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_key",
        "sample_count",
        "gt_non_empty_sample_count",
        "gt_non_empty_ratio",
        "informative_subset_count",
        "informative_pred_non_empty_count",
        "informative_pred_non_empty_ratio",
        "empty_empty_count",
        "empty_empty_ratio",
        "mean_pathway_jaccard_all",
        "mean_pathway_jaccard_informative",
        "mean_direction_consistency_informative",
        "mean_top_10_overlap_informative",
    ]
    if sample_metrics_df.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, object]] = []
    for model_key, group_df in sample_metrics_df.groupby("model_key", sort=False):
        gt_non_empty = pd.to_numeric(group_df.get("gt_total_count", 0), errors="coerce").fillna(0) > 0
        pred_non_empty = pd.to_numeric(group_df.get("pred_total_count", 0), errors="coerce").fillna(0) > 0
        informative = gt_non_empty
        info_pred_non_empty = informative & pred_non_empty
        empty_empty = (~gt_non_empty) & (~pred_non_empty)
        rows.append(
            {
                "model_key": str(model_key),
                "sample_count": int(len(group_df)),
                "gt_non_empty_sample_count": int(gt_non_empty.sum()),
                "gt_non_empty_ratio": float(gt_non_empty.mean()) if len(group_df) else 0.0,
                "informative_subset_count": int(informative.sum()),
                "informative_pred_non_empty_count": int(info_pred_non_empty.sum()),
                "informative_pred_non_empty_ratio": float(
                    info_pred_non_empty.sum() / informative.sum()
                ) if int(informative.sum()) > 0 else 0.0,
                "empty_empty_count": int(empty_empty.sum()),
                "empty_empty_ratio": float(empty_empty.mean()) if len(group_df) else 0.0,
                "mean_pathway_jaccard_all": float(pd.to_numeric(group_df.get("pathway_jaccard", 0), errors="coerce").fillna(0).mean()),
                "mean_pathway_jaccard_informative": float(
                    pd.to_numeric(group_df.loc[informative, "pathway_jaccard"], errors="coerce").fillna(0).mean()
                ) if int(informative.sum()) > 0 and "pathway_jaccard" in group_df.columns else np.nan,
                "mean_direction_consistency_informative": float(
                    pd.to_numeric(group_df.loc[informative, "direction_consistency"], errors="coerce").fillna(0).mean()
                ) if int(informative.sum()) > 0 and "direction_consistency" in group_df.columns else np.nan,
                "mean_top_10_overlap_informative": float(
                    pd.to_numeric(group_df.loc[informative, "top_10_overlap"], errors="coerce").fillna(0).mean()
                ) if int(informative.sum()) > 0 and "top_10_overlap" in group_df.columns else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_moa_metrics(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if sample_metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "moa",
                "model_key",
                "sample_count",
                "mean_pathway_jaccard",
                "mean_direction_consistency",
                "mean_unsupported_extra_rate",
                "mean_null_adjusted_unsupported_extra_rate",
                "mean_top_5_overlap",
                "mean_top_10_overlap",
                "mean_top_20_overlap",
                "mean_knowledge_recall",
            ]
        )

    work_df = sample_metrics_df.copy()
    for metric_col in [
        "null_adjusted_unsupported_extra_rate",
        "top_5_overlap",
        "top_10_overlap",
        "top_20_overlap",
    ]:
        if metric_col not in work_df.columns:
            work_df[metric_col] = 0.0

    return (
        work_df.groupby(["moa", "model_key"], as_index=False)
        .agg(
            sample_count=("sample_id", "nunique") if "sample_id" in work_df.columns else ("model_key", "size"),
            mean_pathway_jaccard=("pathway_jaccard", "mean"),
            mean_direction_consistency=("direction_consistency", "mean"),
            mean_unsupported_extra_rate=("unsupported_extra_rate", "mean"),
            mean_null_adjusted_unsupported_extra_rate=("null_adjusted_unsupported_extra_rate", "mean"),
            mean_top_5_overlap=("top_5_overlap", "mean"),
            mean_top_10_overlap=("top_10_overlap", "mean"),
            mean_top_20_overlap=("top_20_overlap", "mean"),
            mean_knowledge_recall=("knowledge_recall", "mean"),
        )
        .sort_values(["moa", "model_key"])
        .reset_index(drop=True)
    )


def summarize_topk_stability(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "model_key",
        "sample_count",
        "topk_count",
        "mean_pathway_jaccard_std",
        "mean_direction_consistency_std",
    ]
    if sample_metrics_df.empty or "top_k_genes" not in sample_metrics_df.columns:
        return pd.DataFrame(columns=expected_columns)

    per_sample_df = (
        sample_metrics_df.groupby(["model_key", "sample_id"], as_index=False)
        .agg(
            topk_count=("top_k_genes", "nunique"),
            pathway_jaccard_std=("pathway_jaccard", lambda s: float(pd.Series(s).std(ddof=0))),
            direction_consistency_std=("direction_consistency", lambda s: float(pd.Series(s).std(ddof=0))),
        )
    )
    return (
        per_sample_df.groupby("model_key", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            topk_count=("topk_count", "max"),
            mean_pathway_jaccard_std=("pathway_jaccard_std", "mean"),
            mean_direction_consistency_std=("direction_consistency_std", "mean"),
        )
        .sort_values("model_key")
        .reset_index(drop=True)
    )


def summarize_database_robustness(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "model_key",
        "sample_count",
        "database_count",
        "mean_pathway_jaccard_std_across_db",
        "mean_direction_consistency_std_across_db",
    ]
    if sample_metrics_df.empty or "gene_set_db" not in sample_metrics_df.columns:
        return pd.DataFrame(columns=expected_columns)

    per_sample_df = (
        sample_metrics_df.groupby(["model_key", "sample_id"], as_index=False)
        .agg(
            database_count=("gene_set_db", "nunique"),
            pathway_jaccard_std_across_db=("pathway_jaccard", lambda s: float(pd.Series(s).std(ddof=0))),
            direction_consistency_std_across_db=("direction_consistency", lambda s: float(pd.Series(s).std(ddof=0))),
        )
    )
    return (
        per_sample_df.groupby("model_key", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            database_count=("database_count", "max"),
            mean_pathway_jaccard_std_across_db=("pathway_jaccard_std_across_db", "mean"),
            mean_direction_consistency_std_across_db=("direction_consistency_std_across_db", "mean"),
        )
        .sort_values("model_key")
        .reset_index(drop=True)
    )


def summarize_knowledge_support_by_moa(sample_metrics_df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "moa",
        "knowledge_term_count",
        "sample_count",
        "gt_mean_knowledge_recall",
        "best_model_mean_knowledge_recall",
        "knowledge_coverage_flag",
    ]
    if sample_metrics_df.empty:
        return pd.DataFrame(columns=expected_columns)

    work_df = sample_metrics_df.copy()
    work_df["knowledge_term_count"] = pd.to_numeric(
        work_df.get("knowledge_term_count", 0), errors="coerce"
    ).fillna(0)
    work_df["knowledge_recall"] = pd.to_numeric(
        work_df.get("knowledge_recall", 0.0), errors="coerce"
    ).fillna(0.0)

    sample_count_df = (
        work_df.groupby("moa", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            knowledge_term_count=("knowledge_term_count", "max"),
        )
    )
    gt_df = (
        work_df.loc[work_df["model_key"] == "ground_truth"]
        .groupby("moa", as_index=False)
        .agg(gt_mean_knowledge_recall=("knowledge_recall", "mean"))
    )
    model_df = (
        work_df.loc[work_df["model_key"] != "ground_truth"]
        .groupby(["moa", "model_key"], as_index=False)
        .agg(model_mean_knowledge_recall=("knowledge_recall", "mean"))
    )
    best_model_df = (
        model_df.sort_values(["moa", "model_mean_knowledge_recall"], ascending=[True, False])
        .drop_duplicates("moa", keep="first")
        .rename(
            columns={
                "model_key": "best_model_key",
                "model_mean_knowledge_recall": "best_model_mean_knowledge_recall",
            }
        )
    )
    summary_df = sample_count_df.merge(gt_df, on="moa", how="left").merge(best_model_df, on="moa", how="left")
    summary_df["gt_mean_knowledge_recall"] = pd.to_numeric(
        summary_df.get("gt_mean_knowledge_recall", 0.0), errors="coerce"
    ).fillna(0.0)
    summary_df["best_model_mean_knowledge_recall"] = pd.to_numeric(
        summary_df.get("best_model_mean_knowledge_recall", 0.0), errors="coerce"
    ).fillna(0.0)
    summary_df["knowledge_coverage_flag"] = (summary_df["knowledge_term_count"] > 0).astype(int)
    return summary_df.sort_values(["knowledge_coverage_flag", "gt_mean_knowledge_recall", "moa"], ascending=[False, False, True]).reset_index(drop=True)


def build_knowledge_gap_candidate_table(
    sample_metrics_df: pd.DataFrame,
    min_sample_count: int = 1,
) -> pd.DataFrame:
    expected_columns = [
        "moa",
        "sample_count",
        "knowledge_term_count",
        "gt_mean_pathway_count",
        "gap_priority_score",
        "needs_knowledge_curation",
    ]
    if sample_metrics_df.empty:
        return pd.DataFrame(columns=expected_columns)

    work_df = sample_metrics_df.copy()
    work_df["knowledge_term_count"] = pd.to_numeric(work_df.get("knowledge_term_count", 0), errors="coerce").fillna(0)
    work_df["gt_total_count"] = pd.to_numeric(work_df.get("gt_total_count", 0), errors="coerce").fillna(0.0)
    sample_df = (
        work_df.groupby("moa", as_index=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            knowledge_term_count=("knowledge_term_count", "max"),
        )
    )
    gt_df = (
        work_df.loc[work_df["model_key"] == "ground_truth"]
        .groupby("moa", as_index=False)
        .agg(gt_mean_pathway_count=("gt_total_count", "mean"))
    )
    summary_df = sample_df.merge(gt_df, on="moa", how="left")
    summary_df["gt_mean_pathway_count"] = pd.to_numeric(
        summary_df.get("gt_mean_pathway_count", 0.0), errors="coerce"
    ).fillna(0.0)
    summary_df["needs_knowledge_curation"] = (summary_df["knowledge_term_count"] <= 0).astype(int)
    summary_df["gap_priority_score"] = (
        summary_df["needs_knowledge_curation"] * summary_df["sample_count"] * summary_df["gt_mean_pathway_count"]
    )
    summary_df = summary_df.loc[summary_df["sample_count"] >= int(min_sample_count)].copy()
    return summary_df.sort_values(
        ["needs_knowledge_curation", "gap_priority_score", "sample_count", "gt_mean_pathway_count", "moa"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def summarize_moa_pathway_stability(details_df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "moa",
        "model_key",
        "pathway_name",
        "sample_support",
        "sample_support_rate",
    ]
    if details_df.empty:
        return pd.DataFrame(columns=expected_columns)

    dedup_df = details_df.drop_duplicates(["sample_id", "moa", "model_key", "pathway_name"]).copy()
    support_df = (
        dedup_df.groupby(["moa", "model_key", "pathway_name"], as_index=False)
        .agg(sample_support=("sample_id", "nunique"))
    )
    total_df = (
        dedup_df.groupby(["moa", "model_key"], as_index=False)
        .agg(total_samples=("sample_id", "nunique"))
    )
    merged_df = support_df.merge(total_df, on=["moa", "model_key"], how="left")
    merged_df["sample_support_rate"] = (
        merged_df["sample_support"] / merged_df["total_samples"].replace(0, np.nan)
    ).fillna(0.0)
    return (
        merged_df.sort_values(["moa", "model_key", "sample_support", "pathway_name"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )


def build_high_confidence_pathway_table(
    details_df: pd.DataFrame,
    moa_stability_df: pd.DataFrame,
    null_df: pd.DataFrame,
    stability_threshold: float = 0.5,
    null_threshold: float = 0.2,
) -> pd.DataFrame:
    expected_columns = [
        "sample_id",
        "model_key",
        "pathway_name",
        "high_confidence_true_pathway",
    ]
    if details_df.empty:
        return pd.DataFrame(columns=expected_columns)

    label_df = details_df.copy()
    stability_cols = ["moa", "model_key", "pathway_name", "sample_support", "sample_support_rate"]
    null_cols = ["pathway_name", "null_frequency"]

    if not moa_stability_df.empty:
        label_df = label_df.merge(
            moa_stability_df.loc[:, [col for col in stability_cols if col in moa_stability_df.columns]],
            on=["moa", "model_key", "pathway_name"],
            how="left",
        )
    else:
        label_df["sample_support"] = 0
        label_df["sample_support_rate"] = 0.0

    if not null_df.empty:
        label_df = label_df.merge(
            null_df.loc[:, [col for col in null_cols if col in null_df.columns]],
            on="pathway_name",
            how="left",
        )
    else:
        label_df["null_frequency"] = 0.0

    label_df["sample_support"] = pd.to_numeric(label_df.get("sample_support", 0), errors="coerce").fillna(0).astype(int)
    label_df["sample_support_rate"] = pd.to_numeric(label_df.get("sample_support_rate", 0.0), errors="coerce").fillna(0.0)
    label_df["null_frequency"] = pd.to_numeric(label_df.get("null_frequency", 0.0), errors="coerce").fillna(0.0)
    label_df["is_in_gt"] = label_df["is_in_gt"].astype(bool)
    label_df["is_knowledge_supported"] = label_df["is_knowledge_supported"].astype(bool)
    label_df["direction_match_gt"] = label_df["direction_match_gt"].astype(bool)
    label_df["is_moa_stable"] = label_df["sample_support_rate"] >= float(stability_threshold)
    label_df["is_low_null_pathway"] = label_df["null_frequency"] <= float(null_threshold)
    label_df["high_confidence_true_pathway"] = (
        label_df["is_in_gt"]
        & label_df["direction_match_gt"]
        & label_df["is_low_null_pathway"]
        & (label_df["is_knowledge_supported"] | label_df["is_moa_stable"])
    ).astype(int)

    label_df["truth_tier"] = np.where(
        ~label_df["is_in_gt"],
        "not_in_gt",
        np.where(
            label_df["direction_match_gt"]
            & label_df["is_knowledge_supported"]
            & label_df["is_moa_stable"]
            & label_df["is_low_null_pathway"],
            "high_confidence",
            np.where(
                label_df["direction_match_gt"] & label_df["is_knowledge_supported"] & label_df["is_moa_stable"],
                "gt_plus_knowledge_and_stable",
                np.where(
                    label_df["direction_match_gt"] & label_df["is_knowledge_supported"],
                    "gt_plus_knowledge",
                    np.where(
                        label_df["direction_match_gt"] & label_df["is_moa_stable"],
                        "gt_plus_stable",
                        "gt_only",
                    ),
                ),
            ),
        ),
    )
    return label_df.reset_index(drop=True)


def summarize_truth_tiers(high_confidence_df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = ["model_key", "truth_tier", "pathway_count", "pathway_rate"]
    if high_confidence_df.empty:
        return pd.DataFrame(columns=expected_columns)

    work_df = high_confidence_df.copy()
    if "truth_tier" not in work_df.columns:
        work_df["truth_tier"] = np.where(
            pd.to_numeric(work_df.get("high_confidence_true_pathway", 0), errors="coerce").fillna(0).astype(int) > 0,
            "high_confidence",
            "gt_only",
        )
    count_df = (
        work_df.groupby(["model_key", "truth_tier"], as_index=False)
        .agg(pathway_count=("truth_tier", "size"))
    )
    total_df = (
        work_df.groupby("model_key", as_index=False)
        .agg(total_pathway_count=("truth_tier", "size"))
    )
    summary_df = count_df.merge(total_df, on="model_key", how="left")
    summary_df["pathway_rate"] = (
        summary_df["pathway_count"] / summary_df["total_pathway_count"].replace(0, np.nan)
    ).fillna(0.0)
    return summary_df.sort_values(["model_key", "pathway_count", "truth_tier"], ascending=[True, False, True]).reset_index(drop=True)


def select_case_study_samples(sample_metrics_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    if sample_metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "smiles",
                "moa",
                "gt_total_count",
                "baseline_multi",
                "dynamics_modulation",
                "mvcpert",
                "gap_dyn_minus_base",
                "gap_mvcpert_minus_base",
            ]
        )

    sample_metrics_df = sample_metrics_df.copy()
    sample_metrics_df["gt_total_count"] = pd.to_numeric(sample_metrics_df.get("gt_total_count", 0), errors="coerce").fillna(0)
    sample_metrics_df = sample_metrics_df.loc[sample_metrics_df["gt_total_count"] > 0].copy()
    if sample_metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "smiles",
                "moa",
                "gt_total_count",
                "baseline_multi",
                "dynamics_modulation",
                "mvcpert",
                "gap_dyn_minus_base",
                "gap_mvcpert_minus_base",
            ]
        )
    if "knowledge_recall" not in sample_metrics_df.columns:
        sample_metrics_df["knowledge_recall"] = 0.0
    if "knowledge_term_count" not in sample_metrics_df.columns:
        sample_metrics_df["knowledge_term_count"] = 0
    inferred_counts = sample_metrics_df["moa"].astype(str).map(
        lambda moa: len(get_knowledge_terms_for_moa(moa, knowledge_base=load_default_moa_knowledge_base()))
    )
    sample_metrics_df["knowledge_term_count"] = np.maximum(
        pd.to_numeric(sample_metrics_df["knowledge_term_count"], errors="coerce").fillna(0).to_numpy(),
        inferred_counts.to_numpy(),
    )

    pivot_df = (
        sample_metrics_df.pivot_table(
            index=["sample_id", "smiles", "moa", "gt_total_count"],
            columns="model_key",
            values="pathway_jaccard",
            aggfunc="mean",
        )
        .reset_index()
    )
    knowledge_df = (
        sample_metrics_df.groupby(["sample_id", "smiles", "moa", "gt_total_count"], as_index=False)
        .agg(
            mean_knowledge_recall=("knowledge_recall", "mean"),
            mean_knowledge_term_count=("knowledge_term_count", "mean"),
        )
    )
    pivot_df = pivot_df.merge(knowledge_df, on=["sample_id", "smiles", "moa", "gt_total_count"], how="left")
    for required_col in ["baseline_multi", "dynamics_modulation", "mvcpert", "ground_truth"]:
        if required_col not in pivot_df.columns:
            pivot_df[required_col] = 0.0
    pivot_df["gap_dyn_minus_base"] = pivot_df["dynamics_modulation"] - pivot_df["baseline_multi"]
    pivot_df["gap_mvcpert_minus_base"] = pivot_df["mvcpert"] - pivot_df["baseline_multi"]
    pivot_df["signal_priority"] = pivot_df["gt_total_count"].fillna(0).astype(float)
    pivot_df["has_knowledge_mapping"] = (pivot_df["mean_knowledge_term_count"].fillna(0.0) > 0).astype(int)
    pivot_df["has_knowledge_support"] = (pivot_df["mean_knowledge_recall"].fillna(0.0) > 0).astype(int)
    return (
        pivot_df.sort_values(
            [
                "has_knowledge_mapping",
                "has_knowledge_support",
                "mean_knowledge_recall",
                "signal_priority",
                "gap_mvcpert_minus_base",
                "gap_dyn_minus_base",
                "mvcpert",
                "dynamics_modulation",
                "baseline_multi",
            ],
            ascending=[False, False, False, False, False, False, False, False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def compute_null_pathway_frequency(
    gene_symbols: np.ndarray,
    gene_sets: Mapping[str, Set[str]],
    top_k: int = TOP_K_GENES,
    n_iter: int = 200,
    random_seed: int = BENCHMARK_RANDOM_SEED,
    min_hits: int = 2,
) -> pd.DataFrame:
    rng = np.random.RandomState(random_seed)
    all_terms: List[str] = []
    top_k = max(1, min(int(top_k), len(gene_symbols)))
    gene_symbols = np.asarray(gene_symbols)
    background = gene_symbols.tolist()
    for _ in range(int(n_iter)):
        sampled = rng.choice(gene_symbols, size=top_k, replace=False).tolist()
        enr_df = run_ora(sampled, background, gene_sets)
        if enr_df.empty:
            continue
        all_terms.extend(enr_df["Term"].astype(str).tolist())

    if not all_terms:
        return pd.DataFrame(columns=["pathway_name", "null_hit_count", "null_frequency"])

    count_df = pd.Series(all_terms).value_counts().rename_axis("pathway_name").reset_index(name="null_hit_count")
    count_df["null_frequency"] = count_df["null_hit_count"] / max(int(n_iter), 1)
    return count_df.loc[count_df["null_hit_count"] >= int(min_hits)].reset_index(drop=True)


def build_sample_pathway_details(
    benchmark_df: pd.DataFrame,
    gene_symbols: np.ndarray,
    gene_sets: Mapping[str, Set[str]],
    profile_paths: Mapping[str, Path],
    model_keys: Sequence[str] | None = None,
    knowledge_base: Mapping[str, Set[str]] | None = None,
    null_frequency_map: Mapping[str, float] | None = None,
    top_k: int = TOP_K_GENES,
    qvalue_cutoff: float = QVALUE_CUTOFF,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    knowledge_base = knowledge_base or load_default_moa_knowledge_base()
    null_frequency_map = null_frequency_map or {}
    ordered_model_keys = list(model_keys or [k for k in MODEL_ORDER if k != "ground_truth"])
    ordered_model_keys = ordered_model_keys + ["ground_truth"]
    gt_data = load_subset_data_dict("ground_truth", benchmark_df, profile_paths=profile_paths)
    gt_cache: Dict[int, Dict[str, pd.DataFrame]] = {}
    for sample_idx in range(len(benchmark_df)):
        gt_cache[sample_idx] = enrich_signature(
            gt_data["delta_ge"][sample_idx],
            gene_symbols,
            gene_sets,
            top_k=top_k,
            qvalue_cutoff=qvalue_cutoff,
        )

    sample_metric_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    for model_key in ordered_model_keys:
        model_data = gt_data if model_key == "ground_truth" else load_subset_data_dict(
            model_key, benchmark_df, profile_paths=profile_paths
        )
        for sample_idx, row in benchmark_df.reset_index(drop=True).iterrows():
            pred_enrichment = gt_cache[sample_idx] if model_key == "ground_truth" else enrich_signature(
                model_data["delta_ge"][sample_idx],
                gene_symbols,
                gene_sets,
                top_k=top_k,
                qvalue_cutoff=qvalue_cutoff,
            )
            gt_enrichment = gt_cache[sample_idx]

            gt_up = set(gt_enrichment["up"]["Term"].tolist())
            gt_down = set(gt_enrichment["down"]["Term"].tolist())
            pred_up = set(pred_enrichment["up"]["Term"].tolist())
            pred_down = set(pred_enrichment["down"]["Term"].tolist())
            gt_ranked_terms = extract_ranked_pathway_terms(gt_enrichment)
            pred_ranked_terms = extract_ranked_pathway_terms(pred_enrichment)

            metric_row = {
                "sample_id": int(row["sample_index"]),
                "smiles": str(row["smiles"]),
                "moa": str(row["moa"]),
                "model_key": model_key,
                "model": get_model_label(model_key),
                "top_k_genes": int(top_k),
            }
            knowledge_terms = get_knowledge_terms_for_moa(str(row["moa"]), knowledge_base=knowledge_base)
            metric_row.update(
                compare_sample_pathways(
                    gt_up,
                    gt_down,
                    pred_up,
                    pred_down,
                    knowledge_terms=knowledge_terms,
                    null_frequency_map=null_frequency_map,
                    gt_ranked_terms=gt_ranked_terms,
                    pred_ranked_terms=pred_ranked_terms,
                )
            )
            sample_metric_rows.append(metric_row)

            for direction, enr_df in pred_enrichment.items():
                for _, enr_row in enr_df.iterrows():
                    detail_rows.append(
                        {
                            "sample_id": int(row["sample_index"]),
                            "smiles": str(row["smiles"]),
                            "moa": str(row["moa"]),
                            "model_key": model_key,
                            "model": get_model_label(model_key),
                            "direction": direction,
                            "pathway_name": str(enr_row["Term"]),
                            "adjusted_p_value": float(enr_row["Adjusted P-value"]),
                            "odds_ratio": float(enr_row["Odds Ratio"]),
                            "overlap": str(enr_row["Overlap"]),
                            "is_in_gt": str(enr_row["Term"]) in (gt_up | gt_down),
                            "is_knowledge_supported": str(enr_row["Term"]) in knowledge_terms,
                            "direction_match_gt": (
                                (direction == "up" and str(enr_row["Term"]) in gt_up)
                                or (direction == "down" and str(enr_row["Term"]) in gt_down)
                            ),
                        }
                    )

    sample_metrics_df = pd.DataFrame(sample_metric_rows)
    details_df = pd.DataFrame(detail_rows)
    return sample_metrics_df, details_df


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the independent BBBC047 pathway benchmark.")
    parser.add_argument("--baseline-profile", type=Path, default=DEFAULT_PROFILE_PATHS["baseline_multi"])
    parser.add_argument("--dynamics-profile", type=Path, default=DEFAULT_PROFILE_PATHS["dynamics_modulation"])
    parser.add_argument("--mvcpert-profile", type=Path, default=DEFAULT_PROFILE_PATHS["mvcpert"])
    parser.add_argument("--top-k-genes", type=int, default=TOP_K_GENES)
    parser.add_argument(
        "--top-k-grid",
        type=str,
        default="40,80,120",
        help="Comma-separated top-k settings used to summarize pathway stability.",
    )
    parser.add_argument(
        "--gene-set-dbs",
        type=str,
        default="kegg,reactome",
        help="Comma-separated gene set databases to run. Supported: kegg, reactome.",
    )
    parser.add_argument("--qvalue-cutoff", type=float, default=QVALUE_CUTOFF)
    parser.add_argument("--batch-max-per-moa", type=int, default=BATCH_MAX_PER_MOA)
    parser.add_argument("--null-iter", type=int, default=200)
    parser.add_argument("--null-min-hits", type=int, default=2)
    parser.add_argument(
        "--merged-key",
        type=str,
        default=DEFAULT_MERGED_KEY,
        choices=["canon_iso", "canon_noiso", "inchikey14"],
    )
    return parser.parse_args()


def parse_topk_grid(top_k_genes: int, top_k_grid: str) -> List[int]:
    values: List[int] = []
    for raw_value in str(top_k_grid).split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        values.append(int(raw_value))
    values.append(int(top_k_genes))
    return sorted(set(value for value in values if value > 0))


def parse_gene_set_db_list(gene_set_dbs: str) -> List[str]:
    available = get_default_gene_set_configs()
    values = [value.strip().lower() for value in str(gene_set_dbs).split(",") if value.strip()]
    if not values:
        values = ["kegg"]
    unsupported = [value for value in values if value not in available]
    if unsupported:
        raise ValueError(f"Unsupported gene set databases: {unsupported}. Available: {sorted(available)}")
    return values


def main() -> Dict[str, str]:
    args = parse_args()
    profile_paths = {
        "baseline_multi": Path(args.baseline_profile),
        "dynamics_modulation": Path(args.dynamics_profile),
        "mvcpert": Path(args.mvcpert_profile),
    }
    gene_symbols = load_gene_symbols()
    knowledge_base = load_default_moa_knowledge_base()
    moa_index_df = build_moa_index_table_from_merged(
        profile_paths["dynamics_modulation"], key_type=args.merged_key
    )
    benchmark_df = select_benchmark_subset(
        moa_index_df,
        batch_max_per_moa=args.batch_max_per_moa,
    )
    benchmark_df, benchmark_alignment_stats = restrict_benchmark_to_common_keys(
        benchmark_df=benchmark_df,
        profile_paths=profile_paths,
        key_type=args.merged_key,
    )
    if benchmark_df.empty:
        raise ValueError(f"Benchmark subset is empty after common-key restriction: {benchmark_alignment_stats}")
    print(f"Benchmark alignment stats: {benchmark_alignment_stats}")
    gene_set_configs = get_default_gene_set_configs()
    gene_set_db_list = parse_gene_set_db_list(args.gene_set_dbs)
    primary_gene_set_db = gene_set_db_list[0]
    primary_gene_sets = load_kegg_gene_sets(gene_set_configs[primary_gene_set_db])
    null_df = compute_null_pathway_frequency(
        gene_symbols=gene_symbols,
        gene_sets=primary_gene_sets,
        top_k=args.top_k_genes,
        n_iter=args.null_iter,
        min_hits=args.null_min_hits,
    )
    null_frequency_map = (
        null_df.set_index("pathway_name")["null_frequency"].astype(float).to_dict()
        if not null_df.empty and {"pathway_name", "null_frequency"}.issubset(null_df.columns)
        else {}
    )
    sample_metrics_df, details_df = build_sample_pathway_details(
        benchmark_df=benchmark_df,
        gene_symbols=gene_symbols,
        gene_sets=primary_gene_sets,
        profile_paths=profile_paths,
        knowledge_base=knowledge_base,
        null_frequency_map=null_frequency_map,
        top_k=args.top_k_genes,
        qvalue_cutoff=args.qvalue_cutoff,
    )
    sample_metrics_df["gene_set_db"] = primary_gene_set_db
    details_df["gene_set_db"] = primary_gene_set_db
    topk_grid = parse_topk_grid(args.top_k_genes, args.top_k_grid)
    stability_metric_frames: List[pd.DataFrame] = []
    for top_k in topk_grid:
        topk_metrics_df, _ = build_sample_pathway_details(
            benchmark_df=benchmark_df,
            gene_symbols=gene_symbols,
            gene_sets=primary_gene_sets,
            profile_paths=profile_paths,
            knowledge_base=knowledge_base,
            null_frequency_map=null_frequency_map,
            top_k=top_k,
            qvalue_cutoff=args.qvalue_cutoff,
        )
        topk_metrics_df["gene_set_db"] = primary_gene_set_db
        stability_metric_frames.append(topk_metrics_df)
    topk_metric_df = pd.concat(stability_metric_frames, ignore_index=True) if stability_metric_frames else pd.DataFrame()
    database_metric_frames: List[pd.DataFrame] = []
    for gene_set_db in gene_set_db_list:
        gene_sets = load_kegg_gene_sets(gene_set_configs[gene_set_db])
        db_null_df = compute_null_pathway_frequency(
            gene_symbols=gene_symbols,
            gene_sets=gene_sets,
            top_k=args.top_k_genes,
            n_iter=args.null_iter,
            min_hits=args.null_min_hits,
        )
        db_null_frequency_map = (
            db_null_df.set_index("pathway_name")["null_frequency"].astype(float).to_dict()
            if not db_null_df.empty and {"pathway_name", "null_frequency"}.issubset(db_null_df.columns)
            else {}
        )
        db_metrics_df, _ = build_sample_pathway_details(
            benchmark_df=benchmark_df,
            gene_symbols=gene_symbols,
            gene_sets=gene_sets,
            profile_paths=profile_paths,
            knowledge_base=knowledge_base,
            null_frequency_map=db_null_frequency_map,
            top_k=args.top_k_genes,
            qvalue_cutoff=args.qvalue_cutoff,
        )
        db_metrics_df["gene_set_db"] = gene_set_db
        database_metric_frames.append(db_metrics_df)
    database_metrics_df = pd.concat(database_metric_frames, ignore_index=True) if database_metric_frames else pd.DataFrame()
    database_model_summary_frames: List[pd.DataFrame] = []
    for gene_set_db, group_df in database_metrics_df.groupby("gene_set_db", sort=False):
        group_summary_df = summarize_model_metrics(group_df)
        group_summary_df.insert(0, "gene_set_db", gene_set_db)
        database_model_summary_frames.append(group_summary_df)
    database_model_summary_df = (
        pd.concat(database_model_summary_frames, ignore_index=True)
        if database_model_summary_frames
        else pd.DataFrame()
    )
    database_robustness_df = summarize_database_robustness(database_metrics_df)
    knowledge_support_df = summarize_knowledge_support_by_moa(sample_metrics_df)
    knowledge_gap_df = build_knowledge_gap_candidate_table(sample_metrics_df, min_sample_count=1)

    model_summary_df = summarize_model_metrics(sample_metrics_df)
    model_summary_informative_df = summarize_model_metrics_informative(sample_metrics_df)
    moa_summary_df = summarize_moa_metrics(sample_metrics_df)
    moa_summary_informative_df = summarize_moa_metrics_informative(sample_metrics_df)
    signal_audit_df = summarize_signal_audit(sample_metrics_df)
    case_study_df = select_case_study_samples(sample_metrics_df)
    topk_stability_df = summarize_topk_stability(topk_metric_df)
    moa_stability_df = summarize_moa_pathway_stability(details_df)
    high_confidence_df = build_high_confidence_pathway_table(
        details_df=details_df,
        moa_stability_df=moa_stability_df,
        null_df=null_df,
    )
    truth_tier_summary_df = summarize_truth_tiers(high_confidence_df)

    output_paths = {
        "sample_metrics_csv": str(save_csv(sample_metrics_df, "bbbc047_sample_pathway_metrics.csv")),
        "sample_details_csv": str(save_csv(details_df, "bbbc047_sample_pathway_details.csv")),
        "model_summary_csv": str(save_csv(model_summary_df, "bbbc047_model_pathway_summary.csv")),
        "model_summary_informative_csv": str(
            save_csv(model_summary_informative_df, "bbbc047_model_pathway_summary_informative.csv")
        ),
        "moa_summary_csv": str(save_csv(moa_summary_df, "bbbc047_moa_pathway_summary.csv")),
        "moa_summary_informative_csv": str(
            save_csv(moa_summary_informative_df, "bbbc047_moa_pathway_summary_informative.csv")
        ),
        "signal_audit_csv": str(save_csv(signal_audit_df, "bbbc047_signal_audit_summary.csv")),
        "case_study_csv": str(save_csv(case_study_df, "bbbc047_case_study_candidates.csv")),
        "null_summary_csv": str(save_csv(null_df, "bbbc047_null_pathway_frequency.csv")),
        "topk_stability_csv": str(save_csv(topk_stability_df, "bbbc047_topk_stability_summary.csv")),
        "moa_stability_csv": str(save_csv(moa_stability_df, "bbbc047_moa_pathway_stability.csv")),
        "high_confidence_csv": str(save_csv(high_confidence_df, "bbbc047_high_confidence_pathways.csv")),
        "truth_tier_summary_csv": str(save_csv(truth_tier_summary_df, "bbbc047_truth_tier_summary.csv")),
        "database_metrics_csv": str(save_csv(database_metrics_df, "bbbc047_database_sample_pathway_metrics.csv")),
        "database_model_summary_csv": str(save_csv(database_model_summary_df, "bbbc047_database_model_summary.csv")),
        "database_robustness_csv": str(save_csv(database_robustness_df, "bbbc047_database_robustness_summary.csv")),
        "knowledge_support_csv": str(save_csv(knowledge_support_df, "bbbc047_knowledge_support_summary.csv")),
        "knowledge_gap_csv": str(save_csv(knowledge_gap_df, "bbbc047_knowledge_gap_candidates.csv")),
    }
    print(pd.DataFrame([output_paths]).T.to_string(header=False))
    return output_paths


if __name__ == "__main__":
    main()
