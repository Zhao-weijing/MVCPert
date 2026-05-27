#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.pathway_benchmark import bbbc047_pathway_benchmark as bm


DEFAULT_GT_MAP = Path(
    "<ARTIFACT_ROOT>/BBBC036/followups/E21_bbbc036_followups/formal_train/20260417_230448/reference/bbbc036_official_ground_truth_by_smiles.csv"
)
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().with_name("bbbc036_moa_expected_pathways.csv")
DEFAULT_SUMMARY_JSON = Path(__file__).resolve().with_name("bbbc036_moa_expected_pathways_summary.json")
TARGET_SPLIT_RE = re.compile(r"[|,;]")
ALLOWED_PATHWAYS = (
    "Neuroactive ligand-receptor interaction",
    "Calcium signaling pathway",
    "cAMP signaling pathway",
    "cGMP-PKG signaling pathway",
    "Cholinergic synapse",
    "Dopaminergic synapse",
    "Serotonergic synapse",
    "GABAergic synapse",
    "Glutamatergic synapse",
    "Adrenergic signaling in cardiomyocytes",
    "Vascular smooth muscle contraction",
    "Gap junction",
    "Regulation of actin cytoskeleton",
    "MAPK signaling pathway",
    "PI3K-Akt signaling pathway",
    "mTOR signaling pathway",
    "ErbB signaling pathway",
    "FoxO signaling pathway",
    "NF-kappa B signaling pathway",
    "TGF-beta signaling pathway",
    "VEGF signaling pathway",
    "Ras signaling pathway",
    "Rap1 signaling pathway",
    "Focal adhesion",
    "Cell cycle",
    "p53 signaling pathway",
    "Apoptosis",
    "PPAR signaling pathway",
    "Estrogen signaling pathway",
    "Steroid hormone biosynthesis",
    "Ovarian steroidogenesis",
    "Arachidonic acid metabolism",
    "AMPK signaling pathway",
    "Chemokine signaling pathway",
    "Endocytosis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BBBC036-specific MoA-to-pathway prior table.")
    parser.add_argument("--ground-truth-map-csv", type=Path, default=DEFAULT_GT_MAP)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--min-target-overlap", type=int, default=2)
    parser.add_argument("--top-k-per-moa", type=int, default=5)
    return parser.parse_args()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_target_tokens(value: object) -> tuple[str, ...]:
    text = str(value or "")
    tokens: list[str] = []
    for raw_token in TARGET_SPLIT_RE.split(text):
        token = " ".join(raw_token.strip().upper().split())
        if token and token != "NAN":
            tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def main() -> int:
    args = parse_args()
    gt_df = pd.read_csv(args.ground_truth_map_csv)
    gt_df = gt_df.dropna(subset=["Metadata_moa"]).drop_duplicates("canonical_smiles").copy()

    token_support: dict[str, int] = {}
    token_targets: dict[str, set[str]] = {}
    for _, row in gt_df.iterrows():
        moa_tokens = bm.parse_moa_tokens(row["Metadata_moa"])
        if not moa_tokens:
            continue
        targets = set(parse_target_tokens(row.get("Metadata_target", "")))
        for moa_token in moa_tokens:
            token_support[moa_token] = token_support.get(moa_token, 0) + 1
            if targets:
                token_targets.setdefault(moa_token, set()).update(targets)

    gene_sets = bm.load_kegg_gene_sets(bm.KEGG_GMT_PATH)
    allowed_pathways = set(ALLOWED_PATHWAYS)
    rows: list[dict[str, Any]] = []
    matched_token_count = 0
    for moa_keyword, targets in sorted(token_targets.items()):
        candidate_rows: list[tuple[int, str, list[str]]] = []
        for pathway_name, pathway_genes in gene_sets.items():
            if pathway_name not in allowed_pathways:
                continue
            overlap_genes = sorted(targets & pathway_genes)
            if len(overlap_genes) < int(args.min_target_overlap):
                continue
            candidate_rows.append((len(overlap_genes), pathway_name, overlap_genes))
        if not candidate_rows:
            continue
        matched_token_count += 1
        candidate_rows.sort(key=lambda item: (-item[0], item[1]))
        for overlap_count, pathway_name, overlap_genes in candidate_rows[: int(args.top_k_per_moa)]:
            rows.append(
                {
                    "moa_keyword": moa_keyword,
                    "pathway_name": pathway_name,
                    "overlap_count": int(overlap_count),
                    "compound_support": int(token_support.get(moa_keyword, 0)),
                    "target_count": int(len(targets)),
                    "overlap_genes": "|".join(overlap_genes),
                    "source": "bbbc036_target_grounded_allowlist",
                }
            )

    output_df = pd.DataFrame(rows).sort_values(
        ["moa_keyword", "overlap_count", "pathway_name"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    summary = {
        "ground_truth_map_csv": str(args.ground_truth_map_csv),
        "output_csv": str(args.output_csv),
        "min_target_overlap": int(args.min_target_overlap),
        "top_k_per_moa": int(args.top_k_per_moa),
        "allowed_pathway_count": int(len(ALLOWED_PATHWAYS)),
        "official_unique_compound_count": int(len(gt_df)),
        "official_unique_moa_count": int(gt_df["Metadata_moa"].nunique()),
        "matched_token_count": int(matched_token_count),
        "row_count": int(len(output_df)),
        "example_moa_keywords": output_df["moa_keyword"].drop_duplicates().head(20).tolist() if len(output_df) else [],
    }
    json_dump(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
