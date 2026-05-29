#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import inchi
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from umap import UMAP

from paper_plot_style import add_panel_label, apply_publication_style, clean_axes, with_alpha

RDLogger.DisableLog("rdApp.*")
matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_ROOT = SCRIPT_DIR.parent
RELEASE_SOURCE_ROOT = SCRIPT_DIR.parents[3]
OUTPUT_DIR = FIGURE_ROOT / "outputs"
OUTPUT_STEM = "figure5_bbbc047_case_studies"

ARTIFACT_ROOT = Path(os.environ.get("MVCPERT_ARTIFACT_ROOT", "<ARTIFACT_ROOT>"))
AIDD_ROOT = Path(os.environ.get("MVCPERT_AIDD_ROOT", str(RELEASE_SOURCE_ROOT)))
DATA_ROOT = Path(os.environ.get("MVCPERT_DATA_ROOT", "<DATA_ROOT>"))
BASELINE_PROFILE_REL = (
    "2026-04-08/"
    "MVC_splitlock_official_v1_20260408_baseline_augbest40_BBBC047_smiles_split/"
    "ECFP4_Default/predict/test_prediction_profile.h5"
)
MVCPERT_PROFILE_REL = (
    "BBBC047/followups/E40_hypergate_residual_vae_shared_infonce001/"
    "formal_train/20260423_222318/train_outputs/2026-04-23/"
    "MVC_HyperGateResidualVAE_e40_factorized_roles_infonce001_remainder005_officialv1_"
    "BBBC047_smiles_split/ECFP4_Default/predict/test_prediction_profile.h5"
)

BASELINE_PROFILE_PATH = Path(
    os.environ.get(
        "MVCPERT_FIG5_BASELINE_PROFILE_H5",
        str(ARTIFACT_ROOT / BASELINE_PROFILE_REL),
    )
)
MVCPERT_PROFILE_PATH = Path(
    os.environ.get(
        "MVCPERT_FIG5_MVCPERT_PROFILE_H5",
        str(ARTIFACT_ROOT / MVCPERT_PROFILE_REL),
    )
)
DRUG_META_PATH = Path(os.environ.get("MVCPERT_FIG5_DRUG_META_CSV", str(DATA_ROOT / "MVC_BBBC047/merged_repurposing.csv")))
CP_PARQUET_PATH = Path(os.environ.get("MVCPERT_FIG5_CP_PARQUET", str(DATA_ROOT / "MVC_BBBC047/Step2_CP_cleaned.parquet")))
GE_PARQUET_PATH = Path(os.environ.get("MVCPERT_FIG5_GE_PARQUET", str(DATA_ROOT / "MVC_BBBC047/Step2_GE_cleaned.parquet")))
GENE_MAP_PATH = Path(os.environ.get("MVCPERT_FIG5_GENE_MAP_CSV", str(DATA_ROOT / "MVC_BBBC047/gene_map.csv")))
KNOWLEDGE_CSV_PATH = Path(os.environ.get("MVCPERT_FIG5_KNOWLEDGE_CSV", str(AIDD_ROOT / "baseline/pathway_benchmark/knowledge/bbbc047_moa_expected_pathways.csv")))

RANDOM_STATE = 3407
KEY_TYPE_PREFERENCE = ["canon_noiso", "inchikey14", "canon_iso"]
MIN_REPLICATE_WELLS = 8
HEATMAP_REPLICATES = 4
GE_HEATMAP_REPLICATES = 2
TOP_FEATURES_PER_CATEGORY = 3
TOP_FEATURES_FOR_MECHANISM_SCORE = 12
TOP_GE_FEATURES = 15
UMAP_N_NEIGHBORS = 10
CASE_FAMILY_PRIORITY = [
    "CDK inhibitor",
    "EGFR inhibitor",
    "calcium channel blocker",
    "steroid",
    "topoisomerase inhibitor",
    "sodium channel blocker",
    "cyclooxygenase inhibitor",
    "HDAC inhibitor",
    "acetylcholine receptor agonist",
]
CASE_FAMILY_LABELS = {
    "CDK inhibitor": "CDK",
    "topoisomerase inhibitor": "Topo",
    "EGFR inhibitor": "EGFR",
    "calcium channel blocker": "Calcium channel",
    "sodium channel blocker": "Sodium channel",
    "steroid": "Steroid",
    "cyclooxygenase inhibitor": "COX",
    "HDAC inhibitor": "HDAC",
    "acetylcholine receptor agonist": "ACh agonist",
}
FEATURE_CATEGORY_ORDER = [
    "granularity",
    "correlation",
    "spatial distribution",
    "intensity",
    "area shape",
]
FEATURE_CATEGORY_LABELS = {
    "granularity": "Granularity",
    "correlation": "Correlation",
    "spatial distribution": "Spatial distribution",
    "intensity": "Intensity",
    "area shape": "Area shape",
}
CATEGORY_COLOR = {
    "granularity": "#4C78A8",
    "correlation": "#F58518",
    "spatial distribution": "#54A24B",
    "intensity": "#E45756",
    "area shape": "#B279A2",
}
CASE_COLORS = {
    "CDK inhibitor": "#C95D3A",
    "topoisomerase inhibitor": "#A44A7B",
    "EGFR inhibitor": "#2E74B5",
    "calcium channel blocker": "#3B8C59",
    "sodium channel blocker": "#6B8E23",
    "steroid": "#9E6C3B",
    "cyclooxygenase inhibitor": "#7A6AA6",
    "HDAC inhibitor": "#A95AA1",
    "acetylcholine receptor agonist": "#B8860B",
}
MECHANISM_RULES = {
    "CDK inhibitor": {
        "compartments": {"Nuclei", "Cells"},
        "measurements": {"AreaShape", "Granularity", "Intensity", "Texture", "Neighbors"},
        "channels": {"DNA", "RNA", "ER", "AGP"},
    },
    "topoisomerase inhibitor": {
        "compartments": {"Nuclei", "Cells", "Cytoplasm"},
        "measurements": {"AreaShape", "Granularity", "Texture", "Correlation", "Neighbors"},
        "channels": {"DNA", "RNA", "Mito", "AGP"},
    },
    "EGFR inhibitor": {
        "compartments": {"Cells", "Cytoplasm"},
        "measurements": {"RadialDistribution", "Intensity", "Neighbors", "Granularity", "Correlation"},
        "channels": {"RNA", "AGP", "ER", "Mito"},
    },
    "calcium channel blocker": {
        "compartments": {"Cells", "Cytoplasm", "Nuclei"},
        "measurements": {"RadialDistribution", "Intensity", "Granularity", "Correlation", "AreaShape"},
        "channels": {"AGP", "Mito", "RNA", "ER"},
    },
    "sodium channel blocker": {
        "compartments": {"Cells", "Cytoplasm", "Nuclei"},
        "measurements": {"RadialDistribution", "Intensity", "Granularity", "Correlation", "AreaShape"},
        "channels": {"AGP", "RNA", "ER", "Mito"},
    },
    "steroid": {
        "compartments": {"Cells", "Cytoplasm", "Nuclei"},
        "measurements": {"AreaShape", "Texture", "Granularity", "RadialDistribution"},
        "channels": {"ER", "RNA", "AGP", "Mito"},
    },
    "cyclooxygenase inhibitor": {
        "compartments": {"Cytoplasm", "Cells", "Nuclei"},
        "measurements": {"Correlation", "Granularity", "RadialDistribution", "AreaShape"},
        "channels": {"RNA", "ER", "DNA", "AGP"},
    },
    "HDAC inhibitor": {
        "compartments": {"Nuclei", "Cells"},
        "measurements": {"Texture", "Intensity", "AreaShape", "Granularity"},
        "channels": {"DNA", "RNA", "ER", "AGP"},
    },
    "acetylcholine receptor agonist": {
        "compartments": {"Cells", "Cytoplasm", "Nuclei"},
        "measurements": {"RadialDistribution", "Intensity", "Granularity", "Correlation"},
        "channels": {"RNA", "AGP", "ER", "Mito"},
    },
}


@dataclass(frozen=True)
class KeyTypeStats:
    key_type: str
    profile_matched_rows: int
    profile_unique_keys: int
    profile_collision_rows: int
    benchmark_candidate_keys: int


def decode_array(arr: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


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
        key = inchi.MolToInchiKey(mol)
        return key.split("-")[0] if key else None
    raise ValueError(f"Unsupported key type: {key_type}")


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.corrcoef(x, y)[0, 1]
    return float(value) if np.isfinite(value) else float("nan")


def mode_nonempty(series: pd.Series) -> str:
    values = [
        str(v).strip()
        for v in series.dropna().tolist()
        if str(v).strip() and str(v).strip().lower() != "nan"
    ]
    if not values:
        return ""
    vc = pd.Series(values).value_counts()
    return str(vc.index[0])


def parse_cp_feature_name(feature: str) -> tuple[str, str, str]:
    parts = str(feature).split("_")
    compartment = parts[0] if len(parts) > 0 else "Unknown"
    measurement = parts[1] if len(parts) > 1 else "Unknown"
    channel = "Unknown"
    for token in parts:
        if token in {"DNA", "RNA", "ER", "Mito", "AGP"}:
            channel = token
            break
    return compartment, measurement, channel


def cp_feature_category(feature: str) -> str | None:
    _, measurement, _ = parse_cp_feature_name(feature)
    mapping = {
        "Granularity": "granularity",
        "Correlation": "correlation",
        "RadialDistribution": "spatial distribution",
        "Intensity": "intensity",
        "AreaShape": "area shape",
    }
    return mapping.get(measurement)


def shorten_case_name(name: str, width: int = 16) -> str:
    if len(name) <= width:
        return name
    return textwrap.shorten(name, width=width, placeholder="...")


def shorten_feature_name(name: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(name), width=width, break_long_words=False, break_on_hyphens=False))


def load_profiles() -> dict[str, np.ndarray]:
    with h5py.File(BASELINE_PROFILE_PATH, "r") as f_base, h5py.File(MVCPERT_PROFILE_PATH, "r") as f_tm:
        base_smiles = np.asarray(decode_array(f_base["smiles"][:]), dtype=object)
        tm_smiles = np.asarray(decode_array(f_tm["smiles"][:]), dtype=object)
        if base_smiles.shape != tm_smiles.shape or not np.array_equal(base_smiles, tm_smiles):
            raise ValueError("Baseline and MVCPert profile orders do not match.")
        return {
            "smiles": base_smiles,
            "baseline_cp_shift": np.asarray(f_base["cp_pred"][:] - f_base["control_cp"][:], dtype=np.float64),
            "baseline_ge_shift": np.asarray(f_base["ge_pred"][:] - f_base["control_ge"][:], dtype=np.float64),
            "tm_cp_shift": np.asarray(f_tm["cp_pred"][:] - f_tm["control_cp"][:], dtype=np.float64),
            "tm_ge_shift": np.asarray(f_tm["ge_pred"][:] - f_tm["control_ge"][:], dtype=np.float64),
            "cp_true_shift": np.asarray(f_tm["target_cp"][:] - f_tm["control_cp"][:], dtype=np.float64),
            "ge_true_shift": np.asarray(f_tm["target_ge"][:] - f_tm["control_ge"][:], dtype=np.float64),
        }


def load_metadata_with_key(key_type: str) -> pd.DataFrame:
    meta_df = pd.read_csv(DRUG_META_PATH, usecols=["smiles", "pert_iname", "moa", "target"]).copy()
    meta_df["match_key"] = meta_df["smiles"].astype(str).map(lambda s: smiles_key(s, key_type))
    meta_df = meta_df.dropna(subset=["match_key"]).copy()
    grouped = (
        meta_df.groupby("match_key", as_index=False)
        .agg(
            pert_iname=("pert_iname", mode_nonempty),
            moa=("moa", mode_nonempty),
            target=("target", mode_nonempty),
        )
    )
    grouped["key_type"] = key_type
    return grouped[(grouped["pert_iname"] != "") & (grouped["moa"] != "")].copy()


def load_cp_table_with_keys(key_types: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cp_df = pd.read_parquet(CP_PARQUET_PATH).copy()
    feature_cols = [c for c in cp_df.columns if c not in {"SMILES", "pert_dose", "Plate"}]
    smiles_series = cp_df["SMILES"].astype(str)
    for key_type in key_types:
        cp_df[f"match_key_{key_type}"] = smiles_series.map(lambda s: smiles_key(s, key_type))
    return cp_df, feature_cols


def load_ge_table_with_keys(key_types: list[str]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    ge_df = pd.read_parquet(GE_PARQUET_PATH).copy()
    if "SMILES" not in ge_df.columns and "CPD_SMILES" in ge_df.columns:
        ge_df = ge_df.rename(columns={"CPD_SMILES": "SMILES"}).copy()
    if "Plate" not in ge_df.columns and "det_plate" in ge_df.columns:
        ge_df = ge_df.rename(columns={"det_plate": "Plate"}).copy()
    feature_cols = [c for c in ge_df.columns if c not in {"SMILES", "pert_dose", "Plate"}]
    smiles_series = ge_df["SMILES"].astype(str)
    for key_type in key_types:
        ge_df[f"match_key_{key_type}"] = smiles_series.map(lambda s: smiles_key(s, key_type))
    gene_map_df = pd.read_csv(GENE_MAP_PATH).copy()
    return ge_df, feature_cols, gene_map_df


def load_knowledge_base() -> tuple[pd.DataFrame, list[str]]:
    knowledge_df = pd.read_csv(KNOWLEDGE_CSV_PATH).copy()
    keywords = list(dict.fromkeys(knowledge_df["moa_keyword"].astype(str).tolist()))
    return knowledge_df, keywords


def match_knowledge_keywords(moa: str, keywords: list[str]) -> list[str]:
    moa_norm = normalize_text(moa)
    matched = [kw for kw in keywords if normalize_text(kw) in moa_norm]
    return list(dict.fromkeys(matched))


def choose_key_type(
    profile_smiles: np.ndarray,
    cp_df: pd.DataFrame,
    knowledge_keywords: list[str],
) -> tuple[str, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    best_stats: KeyTypeStats | None = None
    best_key_type: str | None = None
    for key_type in KEY_TYPE_PREFERENCE:
        meta_df = load_metadata_with_key(key_type)
        profile_keys = pd.Series([smiles_key(s, key_type) for s in profile_smiles], name="match_key")
        profile_map = pd.DataFrame({"match_key": profile_keys})
        profile_map = profile_map.dropna(subset=["match_key"]).copy()
        matched = profile_map.merge(meta_df[["match_key", "moa"]], on="match_key", how="left")
        matched = matched.dropna(subset=["moa"]).copy()
        profile_collision_rows = int(matched["match_key"].duplicated(keep=False).sum())
        meta_with_knowledge = meta_df.copy()
        meta_with_knowledge["knowledge_matches"] = meta_with_knowledge["moa"].map(
            lambda x: match_knowledge_keywords(str(x), knowledge_keywords)
        )
        meta_with_knowledge = meta_with_knowledge[meta_with_knowledge["knowledge_matches"].map(len) > 0].copy()
        cp_reps = (
            cp_df.dropna(subset=[f"match_key_{key_type}"])
            .groupby(f"match_key_{key_type}")
            .size()
            .rename("replicate_wells")
            .reset_index()
            .rename(columns={f"match_key_{key_type}": "match_key"})
        )
        benchmark_meta = meta_with_knowledge.merge(cp_reps, on="match_key", how="left")
        benchmark_meta["replicate_wells"] = benchmark_meta["replicate_wells"].fillna(0).astype(int)
        benchmark_candidate_keys = int((benchmark_meta["replicate_wells"] >= MIN_REPLICATE_WELLS).sum())
        stats = KeyTypeStats(
            key_type=key_type,
            profile_matched_rows=int(len(matched)),
            profile_unique_keys=int(matched["match_key"].nunique()),
            profile_collision_rows=profile_collision_rows,
            benchmark_candidate_keys=benchmark_candidate_keys,
        )
        summary_rows.append(
            {
                "key_type": stats.key_type,
                "profile_matched_rows": stats.profile_matched_rows,
                "profile_unique_keys": stats.profile_unique_keys,
                "profile_collision_rows": stats.profile_collision_rows,
                "benchmark_candidate_keys": stats.benchmark_candidate_keys,
            }
        )
        if best_stats is None:
            best_stats = stats
            best_key_type = key_type
            continue
        candidate_tuple = (
            stats.benchmark_candidate_keys,
            stats.profile_matched_rows,
            -stats.profile_collision_rows,
            -KEY_TYPE_PREFERENCE.index(stats.key_type),
        )
        best_tuple = (
            best_stats.benchmark_candidate_keys,
            best_stats.profile_matched_rows,
            -best_stats.profile_collision_rows,
            -KEY_TYPE_PREFERENCE.index(best_stats.key_type),
        )
        if candidate_tuple > best_tuple:
            best_stats = stats
            best_key_type = key_type
    if best_key_type is None:
        raise RuntimeError("Failed to choose metadata key type.")
    return best_key_type, pd.DataFrame(summary_rows)


def build_compound_table(
    profile_data: dict[str, np.ndarray],
    key_type: str,
    cp_df: pd.DataFrame,
    feature_cols: list[str],
    knowledge_df: pd.DataFrame,
    knowledge_keywords: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    meta_df = load_metadata_with_key(key_type)
    profile_df = pd.DataFrame(
        {
            "row_index": np.arange(len(profile_data["smiles"]), dtype=int),
            "smiles": profile_data["smiles"].astype(str),
        }
    )
    profile_df["match_key"] = profile_df["smiles"].map(lambda s: smiles_key(s, key_type))
    profile_df = profile_df.dropna(subset=["match_key"]).copy()
    dup_mask = profile_df["match_key"].duplicated(keep=False)
    profile_df = profile_df.loc[~dup_mask].copy()
    compound_df = profile_df.merge(meta_df, on="match_key", how="left")
    compound_df = compound_df.dropna(subset=["moa", "pert_iname"]).copy()
    compound_df["knowledge_matches"] = compound_df["moa"].map(lambda x: match_knowledge_keywords(x, knowledge_keywords))
    compound_df["knowledge_match_count"] = compound_df["knowledge_matches"].map(len)
    compound_df = compound_df[compound_df["knowledge_match_count"] > 0].copy()
    compound_df["primary_knowledge_moa"] = compound_df["knowledge_matches"].map(
        lambda vals: next((family for family in CASE_FAMILY_PRIORITY if family in vals), vals[0])
    )
    expected_pathways = (
        knowledge_df.groupby("moa_keyword")["pathway_name"].apply(lambda s: sorted(set(map(str, s.tolist())))).to_dict()
    )
    compound_df["expected_pathways"] = compound_df["primary_knowledge_moa"].map(expected_pathways)

    cp_key_col = f"match_key_{key_type}"
    cp_reps = cp_df.dropna(subset=[cp_key_col]).groupby(cp_key_col).size().rename("replicate_wells")
    compound_df = compound_df.merge(cp_reps, left_on="match_key", right_index=True, how="left")
    compound_df["replicate_wells"] = compound_df["replicate_wells"].fillna(0).astype(int)
    benchmark_df = compound_df[compound_df["replicate_wells"] >= MIN_REPLICATE_WELLS].copy().reset_index(drop=True)

    dmso_mask = cp_df["SMILES"].astype(str).eq("DMSO")
    dmso_mean = cp_df.loc[dmso_mask, feature_cols].mean()
    dmso_std = cp_df.loc[dmso_mask, feature_cols].std().replace(0.0, np.nan)
    cp_grouped = cp_df.dropna(subset=[cp_key_col]).groupby(cp_key_col, sort=False)

    raw_shift_means: list[np.ndarray] = []
    raw_consistency: list[float] = []
    representative_wells: list[list[int]] = []
    for row in benchmark_df.itertuples(index=False):
        group_df = cp_grouped.get_group(row.match_key).copy()
        shift_mat = group_df.loc[:, feature_cols].to_numpy(dtype=float) - dmso_mean.to_numpy(dtype=float)
        mean_shift = shift_mat.mean(axis=0)
        corr_values = [safe_corr(shift_mat[i], mean_shift) for i in range(shift_mat.shape[0])]
        rep_order = np.argsort(np.nan_to_num(corr_values, nan=-1.0))[::-1]
        representative_wells.append(group_df.index.to_numpy(dtype=int)[rep_order[:HEATMAP_REPLICATES]].tolist())
        raw_shift_means.append(mean_shift)
        raw_consistency.append(float(np.nanmean(corr_values)))

    benchmark_df["raw_mean_shift"] = raw_shift_means
    benchmark_df["raw_consistency"] = raw_consistency
    benchmark_df["representative_well_indices"] = representative_wells
    return benchmark_df, dmso_mean, dmso_std


def compute_sensitive_features(
    benchmark_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
    dmso_std: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    std_arr = dmso_std.to_numpy(dtype=float)
    std_arr = np.where(np.isfinite(std_arr) & (std_arr > 0), std_arr, 1.0)
    row_indices = benchmark_df["row_index"].to_numpy(dtype=int)
    z_shift = profile_data["tm_cp_shift"][row_indices] / std_arr[None, :]
    mean_abs_z = np.mean(np.abs(z_shift), axis=0)
    rows = []
    for idx, feature in enumerate(feature_cols):
        category = cp_feature_category(feature)
        if category is None:
            continue
        rows.append(
            {
                "feature_index": idx,
                "feature": feature,
                "category": category,
                "mean_abs_z": float(mean_abs_z[idx]),
                "compartment": parse_cp_feature_name(feature)[0],
                "measurement": parse_cp_feature_name(feature)[1],
                "channel": parse_cp_feature_name(feature)[2],
            }
        )
    feature_df = pd.DataFrame(rows)
    selected_parts = []
    for category in FEATURE_CATEGORY_ORDER:
        top_part = (
            feature_df[feature_df["category"] == category]
            .sort_values(["mean_abs_z", "feature"], ascending=[False, True], kind="mergesort")
            .head(TOP_FEATURES_PER_CATEGORY)
        )
        selected_parts.append(top_part)
    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df["display_name"] = selected_df["feature"].map(shorten_feature_name)
    selected_df["category_display"] = selected_df["category"].map(FEATURE_CATEGORY_LABELS)
    selected_df["rank_within_category"] = (
        selected_df.groupby("category").cumcount() + 1
    )
    return selected_df


def mechanism_feature_score(
    feature_names: list[str],
    mechanism_family: str,
) -> tuple[float, int]:
    rule = MECHANISM_RULES.get(mechanism_family)
    if rule is None or not feature_names:
        return 0.0, 0
    per_feature_scores = []
    hit_count = 0
    for feature in feature_names:
        compartment, measurement, channel = parse_cp_feature_name(feature)
        score = 0.0
        if compartment in rule["compartments"]:
            score += 1.0
        if measurement in rule["measurements"]:
            score += 1.0
        if channel in rule["channels"]:
            score += 0.5
        if score > 0:
            hit_count += 1
        per_feature_scores.append(score / 2.5)
    return float(np.mean(per_feature_scores)), hit_count


def score_candidate_cases(
    benchmark_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
    selected_features_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    selected_feature_set = set(selected_features_df["feature"].astype(str).tolist())
    rows = []
    for row in benchmark_df.itertuples(index=False):
        idx = int(row.row_index)
        tm_cp_shift = profile_data["tm_cp_shift"][idx]
        base_cp_shift = profile_data["baseline_cp_shift"][idx]
        cp_true_shift = profile_data["cp_true_shift"][idx]
        tm_ge_shift = profile_data["tm_ge_shift"][idx]
        ge_true_shift = profile_data["ge_true_shift"][idx]

        tm_cp_true_corr = safe_corr(tm_cp_shift, cp_true_shift)
        base_cp_true_corr = safe_corr(base_cp_shift, cp_true_shift)
        tm_minus_base_cp = tm_cp_true_corr - base_cp_true_corr if np.isfinite(tm_cp_true_corr) and np.isfinite(base_cp_true_corr) else float("nan")
        tm_ge_true_corr = safe_corr(tm_ge_shift, ge_true_shift)

        order = np.argsort(np.abs(tm_cp_shift))[::-1]
        top_feature_names = [feature_cols[i] for i in order[:TOP_FEATURES_FOR_MECHANISM_SCORE]]
        mechanism_score, mechanism_hit_count = mechanism_feature_score(top_feature_names, str(row.primary_knowledge_moa))
        selected_biomarker_hits = int(sum(feature in selected_feature_set for feature in top_feature_names))

        scaled_cp_gain = np.clip(tm_minus_base_cp / 0.15, 0.0, 1.0) if np.isfinite(tm_minus_base_cp) else 0.0
        case_score = (
            0.30 * mechanism_score
            + 0.20 * np.clip(float(row.raw_consistency), 0.0, 1.0)
            + 0.20 * np.clip(tm_cp_true_corr, 0.0, 1.0)
            + 0.15 * scaled_cp_gain
            + 0.15 * np.clip(tm_ge_true_corr, 0.0, 1.0)
        )

        rows.append(
            {
                "match_key": row.match_key,
                "row_index": idx,
                "pert_iname": row.pert_iname,
                "moa": row.moa,
                "target": row.target,
                "primary_knowledge_moa": row.primary_knowledge_moa,
                "replicate_wells": int(row.replicate_wells),
                "raw_consistency": float(row.raw_consistency),
                "tm_cp_true_corr": tm_cp_true_corr,
                "base_cp_true_corr": base_cp_true_corr,
                "tm_minus_base_cp": tm_minus_base_cp,
                "tm_ge_true_corr": tm_ge_true_corr,
                "mechanism_score": mechanism_score,
                "mechanism_hit_count": mechanism_hit_count,
                "selected_biomarker_hits": selected_biomarker_hits,
                "case_score": float(case_score),
                "top_mechanism_features": " | ".join(top_feature_names[:8]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["case_score", "mechanism_score", "raw_consistency", "tm_cp_true_corr"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def select_representative_cases(candidate_df: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    used_keys: set[str] = set()
    preferred_case_names = {
        "CDK inhibitor": "aminopurvalanol-a",
        "EGFR inhibitor": "tyrphostin-AG-1478",
        "calcium channel blocker": "tetrandrine",
        "steroid": "clocortolone-pivalate",
    }
    for family in CASE_FAMILY_PRIORITY:
        family_df = candidate_df[
            (candidate_df["primary_knowledge_moa"] == family)
            & (~candidate_df["match_key"].isin(used_keys))
        ].copy()
        if family_df.empty:
            continue
        preferred_name = preferred_case_names.get(family)
        if preferred_name is not None:
            preferred_df = family_df[family_df["pert_iname"] == preferred_name].copy()
            row = preferred_df.iloc[0] if not preferred_df.empty else family_df.iloc[0]
        else:
            row = family_df.iloc[0]
        selected_rows.append(row)
        used_keys.add(str(row["match_key"]))
        if len(selected_rows) == 4:
            break
    if len(selected_rows) < 4:
        fallback_df = candidate_df[~candidate_df["match_key"].isin(used_keys)].copy()
        for _, row in fallback_df.iterrows():
            selected_rows.append(row)
            if len(selected_rows) == 4:
                break
    if len(selected_rows) < 4:
        raise RuntimeError("Failed to select four representative mechanism-consistent cases.")
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def compute_same_moa_hit_rate(feature_matrix: np.ndarray, labels: list[str], k: int = 3) -> float:
    if feature_matrix.shape[0] <= 1:
        return float("nan")
    dist = pairwise_distances(feature_matrix, metric="cosine")
    hits = []
    for i in range(feature_matrix.shape[0]):
        order = np.argsort(dist[i])
        order = order[order != i][:k]
        if order.size == 0:
            continue
        hits.append(float(any(labels[j] == labels[i] for j in order)))
    return float(np.mean(hits)) if hits else float("nan")


def build_umap_coords(
    benchmark_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    row_indices = benchmark_df["row_index"].to_numpy(dtype=int)
    base_mat = profile_data["baseline_cp_shift"][row_indices]
    tm_mat = profile_data["tm_cp_shift"][row_indices]
    combined = np.vstack([base_mat, tm_mat])
    n_components = min(30, combined.shape[0] - 1, combined.shape[1])
    pca = PCA(n_components=max(n_components, 2), random_state=RANDOM_STATE)
    pcs = pca.fit_transform(combined)
    n_neighbors = max(4, min(UMAP_N_NEIGHBORS, pcs.shape[0] - 1))
    coords = UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.35,
        metric="euclidean",
        random_state=RANDOM_STATE,
    ).fit_transform(pcs)
    n_rows = len(benchmark_df)
    labels = benchmark_df["primary_knowledge_moa"].astype(str).tolist()
    base_hit = compute_same_moa_hit_rate(base_mat, labels)
    tm_hit = compute_same_moa_hit_rate(tm_mat, labels)
    return coords[:n_rows], coords[n_rows:], base_hit, tm_hit


def select_heatmap_wells(
    benchmark_df: pd.DataFrame,
    selected_cases_df: pd.DataFrame,
    cp_df: pd.DataFrame,
    dmso_mean: pd.Series,
    feature_cols: list[str],
    key_type: str,
) -> pd.DataFrame:
    cp_key_col = f"match_key_{key_type}"
    selected_lookup = benchmark_df.set_index("match_key")
    rows = []
    for row in selected_cases_df.itertuples(index=False):
        benchmark_row = selected_lookup.loc[row.match_key]
        for order, well_index in enumerate(benchmark_row["representative_well_indices"], start=1):
            well_row = cp_df.loc[int(well_index)]
            shift_vec = well_row.loc[feature_cols].to_numpy(dtype=float) - dmso_mean.to_numpy(dtype=float)
            rows.append(
                {
                    "group": benchmark_row["pert_iname"],
                    "group_label": benchmark_row["primary_knowledge_moa"],
                    "column_label": f"{shorten_case_name(benchmark_row['pert_iname'])}\n({order})",
                    "match_key": row.match_key,
                    "source": "treated",
                    "shift": shift_vec,
                }
            )
    selected_plates = cp_df.loc[
        cp_df[cp_key_col].isin(selected_cases_df["match_key"]),
        "Plate",
    ].dropna().unique().tolist()
    dmso_pool = cp_df[cp_df["SMILES"].astype(str).eq("DMSO")].copy()
    if selected_plates:
        plate_filtered = dmso_pool[dmso_pool["Plate"].isin(selected_plates)].copy()
        if len(plate_filtered) >= HEATMAP_REPLICATES:
            dmso_pool = plate_filtered
    dmso_shift = dmso_pool.loc[:, feature_cols].to_numpy(dtype=float) - dmso_mean.to_numpy(dtype=float)[None, :]
    norms = np.linalg.norm(dmso_shift, axis=1)
    keep = np.argsort(norms)[:HEATMAP_REPLICATES]
    for order, (_, well_row) in enumerate(dmso_pool.iloc[keep].iterrows(), start=1):
        rows.insert(
            order - 1,
            {
                "group": "DMSO",
                "group_label": "Control",
                "column_label": f"DMSO\n({order})",
                "match_key": "DMSO",
                "source": "control",
                "shift": well_row.loc[feature_cols].to_numpy(dtype=float) - dmso_mean.to_numpy(dtype=float),
            },
        )
    return pd.DataFrame(rows)


def build_heatmap_matrix(
    heatmap_wells_df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    dmso_std: pd.Series,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str], list[str], list[tuple[int, str]]]:
    feature_index = {feature: idx for idx, feature in enumerate(feature_cols)}
    selected_features = selected_features_df["feature"].astype(str).tolist()
    std_arr = dmso_std.to_numpy(dtype=float)
    std_arr = np.where(np.isfinite(std_arr) & (std_arr > 0), std_arr, 1.0)
    matrix = []
    for row in heatmap_wells_df.itertuples(index=False):
        shift = np.asarray(row.shift, dtype=float)
        z = shift / std_arr
        matrix.append([float(z[feature_index[f]]) for f in selected_features])
    heatmap = np.asarray(matrix, dtype=float).T
    heatmap = np.clip(heatmap, -3.0, 3.0)
    row_labels = selected_features_df["display_name"].astype(str).tolist()
    col_labels = heatmap_wells_df["column_label"].astype(str).tolist()
    groups = []
    start = 0
    for group_name, group_df in heatmap_wells_df.groupby("group", sort=False):
        groups.append((start, str(group_name)))
        start += len(group_df)
    return heatmap, row_labels, col_labels, groups


def compute_sensitive_ge_features(
    benchmark_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
    gene_feature_cols: list[str],
    gene_map_df: pd.DataFrame,
) -> pd.DataFrame:
    row_indices = benchmark_df["row_index"].to_numpy(dtype=int)
    ge_shift = profile_data["tm_ge_shift"][row_indices]
    mean_abs = np.mean(np.abs(ge_shift), axis=0)
    gene_symbol_map = dict(zip(gene_map_df["probe_id"].astype(str), gene_map_df["gene_symbol"].astype(str)))
    feature_df = pd.DataFrame(
        {
            "feature_index": np.arange(len(gene_feature_cols), dtype=int),
            "feature": list(map(str, gene_feature_cols)),
            "mean_abs_shift": mean_abs.astype(float),
        }
    )
    feature_df["gene_symbol"] = feature_df["feature"].map(lambda x: gene_symbol_map.get(x, x))
    selected_df = (
        feature_df.sort_values(["mean_abs_shift", "feature"], ascending=[False, True], kind="mergesort")
        .head(TOP_GE_FEATURES)
        .reset_index(drop=True)
    )
    selected_df["display_name"] = selected_df["gene_symbol"].astype(str)
    selected_df["rank"] = np.arange(1, len(selected_df) + 1, dtype=int)
    return selected_df


def select_ge_heatmap_rows(
    selected_cases_df: pd.DataFrame,
    ge_df: pd.DataFrame,
    gene_feature_cols: list[str],
    key_type: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ge_key_col = f"match_key_{key_type}"
    dmso_mask = ge_df["SMILES"].astype(str).eq("DMSO")
    dmso_mean = ge_df.loc[dmso_mask, gene_feature_cols].mean()
    dmso_std = ge_df.loc[dmso_mask, gene_feature_cols].std().replace(0.0, np.nan)
    rows = []
    for row in selected_cases_df.itertuples(index=False):
        sub = ge_df[ge_df[ge_key_col] == row.match_key].copy()
        if sub.empty:
            continue
        shift_mat = sub.loc[:, gene_feature_cols].to_numpy(dtype=float) - dmso_mean.to_numpy(dtype=float)
        mean_shift = shift_mat.mean(axis=0)
        corr_values = [safe_corr(shift_mat[i], mean_shift) for i in range(shift_mat.shape[0])]
        order = np.argsort(np.nan_to_num(corr_values, nan=-1.0))[::-1][:GE_HEATMAP_REPLICATES]
        for rank, idx in enumerate(order, start=1):
            shift_vec = shift_mat[idx]
            rows.append(
                {
                    "group": row.pert_iname,
                    "column_label": f"{shorten_case_name(str(row.pert_iname))}\n({rank})",
                    "source": "treated",
                    "shift": shift_vec,
                }
            )
    dmso_pool = ge_df[dmso_mask].copy()
    dmso_shift = dmso_pool.loc[:, gene_feature_cols].to_numpy(dtype=float) - dmso_mean.to_numpy(dtype=float)[None, :]
    norms = np.linalg.norm(dmso_shift, axis=1)
    keep = np.argsort(norms)[:GE_HEATMAP_REPLICATES]
    for order, idx in enumerate(keep, start=1):
        rows.insert(
            order - 1,
            {
                "group": "DMSO",
                "column_label": f"DMSO\n({order})",
                "source": "control",
                "shift": dmso_shift[idx],
            },
        )
    return pd.DataFrame(rows), dmso_mean, dmso_std


def build_ge_heatmap_matrix(
    ge_heatmap_df: pd.DataFrame,
    selected_ge_features_df: pd.DataFrame,
    ge_dmso_std: pd.Series,
    gene_feature_cols: list[str],
) -> tuple[np.ndarray, list[str], list[str], list[tuple[int, str]]]:
    feature_index = {feature: idx for idx, feature in enumerate(gene_feature_cols)}
    selected_features = selected_ge_features_df["feature"].astype(str).tolist()
    std_arr = ge_dmso_std.to_numpy(dtype=float)
    std_arr = np.where(np.isfinite(std_arr) & (std_arr > 0), std_arr, 1.0)
    matrix = []
    for row in ge_heatmap_df.itertuples(index=False):
        shift = np.asarray(row.shift, dtype=float)
        z = shift / std_arr
        matrix.append([float(z[feature_index[f]]) for f in selected_features])
    heatmap = np.asarray(matrix, dtype=float).T
    heatmap = np.clip(heatmap, -3.0, 3.0)
    row_labels = selected_ge_features_df["display_name"].astype(str).tolist()
    col_labels = ge_heatmap_df["column_label"].astype(str).tolist()
    groups = []
    start = 0
    for group_name, group_df in ge_heatmap_df.groupby("group", sort=False):
        groups.append((start, str(group_name)))
        start += len(group_df)
    return heatmap, row_labels, col_labels, groups


def plot_umap_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    benchmark_df: pd.DataFrame,
    selected_cases_df: pd.DataFrame,
    title: str,
) -> None:
    families = benchmark_df["primary_knowledge_moa"].astype(str).tolist()
    unique_families = list(dict.fromkeys(families))
    for family in unique_families:
        sub = benchmark_df[benchmark_df["primary_knowledge_moa"] == family].copy()
        idx = sub.index.to_numpy(dtype=int)
        color = CASE_COLORS.get(family, "#6B7280")
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=38,
            color=with_alpha(color, 0.82),
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        if len(sub) >= 2 or family in selected_cases_df["primary_knowledge_moa"].tolist():
            cx = float(np.median(coords[idx, 0]))
            cy = float(np.median(coords[idx, 1]))
            ax.text(
                cx,
                cy,
                CASE_FAMILY_LABELS.get(family, family),
                fontsize=8,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": with_alpha("#FFFFFF", 0.78),
                    "edgecolor": with_alpha(color, 0.6),
                },
                zorder=4,
            )
    for row in selected_cases_df.itertuples(index=False):
        sub = benchmark_df[benchmark_df["match_key"] == row.match_key]
        if sub.empty:
            continue
        idx = int(sub.index[0])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=110,
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.2,
            zorder=5,
        )
        ax.text(
            coords[idx, 0],
            coords[idx, 1] + 0.42,
            shorten_case_name(str(row.pert_iname), width=18),
            fontsize=8,
            ha="center",
            va="bottom",
            zorder=6,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)


def plot_heatmap_panel(
    ax: plt.Axes,
    heatmap: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    groups: list[tuple[int, str]],
    selected_features_df: pd.DataFrame,
) -> None:
    im = ax.imshow(heatmap, aspect="auto", cmap="RdBu", vmin=-3.0, vmax=3.0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)

    for boundary, _ in groups[1:]:
        ax.axvline(boundary - 0.5, color="#FFFFFF", linewidth=1.2)

    category_series = selected_features_df["category"].astype(str).tolist()
    category_boundaries = []
    current = category_series[0]
    for idx, category in enumerate(category_series[1:], start=1):
        if category != current:
            category_boundaries.append(idx)
            current = category
    for boundary in category_boundaries:
        ax.axhline(boundary - 0.5, color="#FFFFFF", linewidth=1.2)

    for start, group_name in groups:
        end = len(col_labels)
        for next_start, _ in groups:
            if next_start > start:
                end = next_start
                break
        center = (start + end - 1) / 2.0
        if group_name == "DMSO":
            label = "DMSO"
        else:
            label = shorten_case_name(group_name, width=16)
        ax.text(
            center,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    category_positions = []
    start = 0
    current = category_series[0]
    for idx, category in enumerate(category_series[1:], start=1):
        if category != current:
            category_positions.append(((start + idx - 1) / 2.0, current))
            start = idx
            current = category
    category_positions.append(((start + len(category_series) - 1) / 2.0, current))
    for pos, category in category_positions:
        ax.text(
            -0.07,
            pos,
            FEATURE_CATEGORY_LABELS.get(category, category),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            color=CATEGORY_COLOR.get(category, "#333333"),
            fontweight="bold",
        )
    ax.set_title("Representative raw Cell Painting wells on 15 MVCPert-selected biomarkers", fontsize=11)
    return im


def plot_case_bars(
    ax: plt.Axes,
    case_row: pd.Series,
    selected_features_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
    dmso_std: pd.Series,
    feature_cols: list[str],
) -> None:
    raise NotImplementedError("Legacy cartesian renderer replaced by radial renderer.")


def radial_bar_colors(signed_values: np.ndarray) -> list[str]:
    colors = []
    for value in signed_values:
        if value < 0:
            colors.append("#9dc1c5")
        elif value > 0:
            colors.append("#d7a6b3")
        else:
            colors.append("#8e93af")
    return colors


def format_polar_tick_label(angle_deg: float) -> tuple[float, str]:
    if 90 < angle_deg < 270:
        return angle_deg + 180, "right"
    return angle_deg, "left"


def plot_radial_case_bars(
    ax: plt.Axes,
    signed_values: np.ndarray,
    labels: list[str],
    title: str,
    subtitle: str,
    label_fontsize: float = 5.1,
    title_fontsize: float = 9.5,
    subtitle_fontsize: float = 7.2,
    label_radius_factor: float = 1.05,
) -> None:
    count = len(labels)
    if count == 0:
        ax.set_axis_off()
        return

    values = np.abs(np.asarray(signed_values, dtype=float))
    local_max = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(local_max) or local_max <= 0:
        local_max = 1.0
    radial_max = local_max
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    width = 2.0 * np.pi / count
    base_radius = max(radial_max * 0.34, 0.8)
    colors = radial_bar_colors(np.asarray(signed_values, dtype=float))

    bars = ax.bar(
        angles,
        values,
        width=width,
        bottom=base_radius,
        color=colors,
        edgecolor="#f1f5f9",
        linewidth=1.0,
        zorder=2,
    )

    for angle, value in zip(angles, values):
        angle_deg = float(np.degrees(angle))
        rotation, ha = format_polar_tick_label(angle_deg)
        text_radius = base_radius + max(value * 0.72, value - max(radial_max * 0.08, 0.18))
        text_obj = ax.text(
            angle,
            text_radius,
            f"{value:.1f}",
            ha=ha,
            va="center",
            fontsize=label_fontsize - 0.1,
            fontweight="bold",
            rotation=rotation,
            rotation_mode="anchor",
            color="#2f3f4f",
            zorder=4,
        )
        text_obj.set_path_effects([pe.withStroke(linewidth=1.0, foreground="#f7f7f7")])

    outer_padding = max(radial_max * 0.12, 0.3)
    ax.set_ylim(0.0, base_radius + radial_max + outer_padding)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=label_fontsize, pad=-5)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", linewidth=0.5, color="#8e93af", alpha=0.5, zorder=1)
    ax.set_facecolor("white")

    ax.text(
        0.5,
        -0.10,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        0.5,
        -0.20,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=subtitle_fontsize,
        color="#555555",
        zorder=6,
    )


def plot_cp_case_radial_bars(
    ax: plt.Axes,
    case_row: pd.Series,
    selected_features_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
    dmso_std: pd.Series,
    feature_cols: list[str],
) -> None:
    std_arr = dmso_std.to_numpy(dtype=float)
    std_arr = np.where(np.isfinite(std_arr) & (std_arr > 0), std_arr, 1.0)
    feature_index = {feature: idx for idx, feature in enumerate(feature_cols)}
    row_index = int(case_row["row_index"])
    pred_z = profile_data["tm_cp_shift"][row_index] / std_arr
    ordered_features = selected_features_df["feature"].astype(str).tolist()
    signed_values = np.array([pred_z[feature_index[feature]] for feature in ordered_features], dtype=float)
    labels = [str(i + 1) for i in range(len(ordered_features))]
    plot_radial_case_bars(
        ax=ax,
        signed_values=signed_values,
        labels=labels,
        title=shorten_case_name(str(case_row["pert_iname"]), width=24),
        subtitle=str(case_row["primary_knowledge_moa"]),
        label_fontsize=5.7,
        title_fontsize=9.0,
        subtitle_fontsize=8.0,
        label_radius_factor=0.95,
    )


def plot_ge_case_bars(
    ax: plt.Axes,
    case_row: pd.Series,
    selected_ge_features_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
    gene_feature_cols: list[str],
) -> None:
    raise NotImplementedError("Legacy cartesian renderer replaced by radial renderer.")


def plot_ge_case_radial_bars(
    ax: plt.Axes,
    case_row: pd.Series,
    selected_ge_features_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
    gene_feature_cols: list[str],
) -> None:
    feature_index = {feature: idx for idx, feature in enumerate(gene_feature_cols)}
    row_index = int(case_row["row_index"])
    pred_values = profile_data["tm_ge_shift"][row_index]
    ordered_features = selected_ge_features_df["feature"].astype(str).tolist()
    signed_values = np.array([pred_values[feature_index[feature]] for feature in ordered_features], dtype=float)
    labels = [str(i + 1) for i in range(len(ordered_features))]
    plot_radial_case_bars(
        ax=ax,
        signed_values=signed_values,
        labels=labels,
        title=shorten_case_name(str(case_row["pert_iname"]), width=24),
        subtitle=str(case_row["primary_knowledge_moa"]),
        label_fontsize=6.1,
        title_fontsize=9.0,
        subtitle_fontsize=8.0,
        label_radius_factor=0.98,
    )


def build_figure(
    benchmark_df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    selected_ge_features_df: pd.DataFrame,
    selected_cases_df: pd.DataFrame,
    profile_data: dict[str, np.ndarray],
    dmso_mean: pd.Series,
    dmso_std: pd.Series,
    cp_df: pd.DataFrame,
    feature_cols: list[str],
    ge_df: pd.DataFrame,
    gene_feature_cols: list[str],
    key_type: str,
) -> None:
    heatmap_wells_df = select_heatmap_wells(benchmark_df, selected_cases_df, cp_df, dmso_mean, feature_cols, key_type)
    heatmap, row_labels, col_labels, groups = build_heatmap_matrix(
        heatmap_wells_df,
        selected_features_df,
        dmso_std,
        feature_cols,
    )
    ge_heatmap_df, ge_dmso_mean, ge_dmso_std = select_ge_heatmap_rows(
        selected_cases_df=selected_cases_df,
        ge_df=ge_df,
        gene_feature_cols=gene_feature_cols,
        key_type=key_type,
    )
    ge_heatmap, ge_row_labels, ge_col_labels, ge_groups = build_ge_heatmap_matrix(
        ge_heatmap_df=ge_heatmap_df,
        selected_ge_features_df=selected_ge_features_df,
        ge_dmso_std=ge_dmso_std,
        gene_feature_cols=gene_feature_cols,
    )

    fig = plt.figure(figsize=(13.2, 7.1))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        wspace=0.08,
    )

    ax_cases_outer = fig.add_subplot(gs[0, 0])
    ax_cases_outer.set_axis_off()
    add_panel_label(ax_cases_outer, "A")
    ax_cases_outer.text(
        0.0,
        1.02,
        "Cell Painting Biomarker Shifts",
        transform=ax_cases_outer.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    inner = gs[0, 0].subgridspec(2, 2, wspace=0.20, hspace=0.26)
    for panel_index, (_, case_row) in enumerate(selected_cases_df.iterrows()):
        ax_case = fig.add_subplot(inner[panel_index // 2, panel_index % 2], projection="polar")
        plot_cp_case_radial_bars(
            ax_case,
            case_row,
            selected_features_df,
            profile_data,
            dmso_std,
            feature_cols,
        )

    ax_ge_cases_outer = fig.add_subplot(gs[0, 1])
    ax_ge_cases_outer.set_axis_off()
    add_panel_label(ax_ge_cases_outer, "B")
    ax_ge_cases_outer.text(
        0.0,
        1.02,
        "Gene-expression Shifts",
        transform=ax_ge_cases_outer.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    inner_ge = gs[0, 1].subgridspec(2, 2, wspace=0.20, hspace=0.26)
    for panel_index, (_, case_row) in enumerate(selected_cases_df.iterrows()):
        ax_case = fig.add_subplot(inner_ge[panel_index // 2, panel_index % 2], projection="polar")
        plot_ge_case_radial_bars(
            ax_case,
            case_row,
            selected_ge_features_df,
            profile_data,
            gene_feature_cols,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    apply_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profile_data = load_profiles()
    knowledge_df, knowledge_keywords = load_knowledge_base()
    cp_df, feature_cols = load_cp_table_with_keys(KEY_TYPE_PREFERENCE)
    ge_df, gene_feature_cols, gene_map_df = load_ge_table_with_keys(KEY_TYPE_PREFERENCE)
    key_type, _ = choose_key_type(profile_data["smiles"], cp_df, knowledge_keywords)

    benchmark_df, dmso_mean, dmso_std = build_compound_table(
        profile_data=profile_data,
        key_type=key_type,
        cp_df=cp_df,
        feature_cols=feature_cols,
        knowledge_df=knowledge_df,
        knowledge_keywords=knowledge_keywords,
    )
    if benchmark_df.empty:
        raise RuntimeError("No benchmark compounds survived the well-characterized MoA + replicate-well filter.")

    selected_features_df = compute_sensitive_features(
        benchmark_df=benchmark_df,
        profile_data=profile_data,
        dmso_std=dmso_std,
        feature_cols=feature_cols,
    )
    candidate_df = score_candidate_cases(
        benchmark_df=benchmark_df,
        profile_data=profile_data,
        selected_features_df=selected_features_df,
        feature_cols=feature_cols,
    )
    selected_ge_features_df = compute_sensitive_ge_features(
        benchmark_df=benchmark_df,
        profile_data=profile_data,
        gene_feature_cols=gene_feature_cols,
        gene_map_df=gene_map_df,
    )
    selected_cases_df = select_representative_cases(candidate_df)

    build_figure(
        benchmark_df=benchmark_df,
        selected_features_df=selected_features_df,
        selected_ge_features_df=selected_ge_features_df,
        selected_cases_df=selected_cases_df,
        profile_data=profile_data,
        dmso_mean=dmso_mean,
        dmso_std=dmso_std,
        cp_df=cp_df,
        feature_cols=feature_cols,
        ge_df=ge_df,
        gene_feature_cols=gene_feature_cols,
        key_type=key_type,
    )


if __name__ == "__main__":
    main()
