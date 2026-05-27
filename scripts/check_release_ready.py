#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".h5",
    ".h5ad",
    ".npz",
    ".pickle",
    ".pkl",
    ".parquet",
    ".pyc",
    ".ipynb",
    ".zip",
    ".gz",
    ".png",
}
UNEXPECTED_PATH_PARTS = {
    "__pycache__",
    "logs",
    "runs",
}
UNEXPECTED_PATH_PREFIXES = {
    "source/paper/",
}
FORBIDDEN_NAMES = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
}
FORBIDDEN_TEXT_PATTERNS = [
    re.compile(r"/home/[A-Za-z0-9_.-]+"),
    re.compile(r"(?<![>}$A-Za-z0-9_.-])/data/[A-Za-z0-9_.-]+"),
]
ALLOWED_TEXT_SNIPPETS = {
    "/path/to/data",
}
REQUIRED_FILES = [
    "README.md",
    "RUNBOOK.md",
    "ENVIRONMENT.md",
    "DATA_AND_ARTIFACTS.md",
    "RELEASE_MANIFEST.json",
    "source/baseline/src/train_mvc.py",
    "source/baseline/src/MVCModel_HyperGate.py",
    "source/baseline/src/model_MVC_residual_vae.py",
    "source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json",
    "source/figures/paper/figure2_multimodal/scripts/generate_figure2_multimodal.py",
    "source/figures/paper/figure3_architecture_ablation/scripts/generate_figure3_architecture_ablation.py",
    "source/figures/paper/figure4_missing_modality_virtual_profiling/scripts/generate_figure4_missing_modality_virtual_profiling.py",
    "source/figures/paper/figure5_case_studies/scripts/generate_figure5_case_studies.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MVCPert release package hygiene.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES)
    ]
    unexpected_paths = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if any(part in UNEXPECTED_PATH_PARTS for part in path.parts)
        or any(path.relative_to(root).as_posix().startswith(prefix) for prefix in UNEXPECTED_PATH_PREFIXES)
    ]
    text_leaks = []
    for path in root.rglob("*"):
        rel_path = path.relative_to(root).as_posix()
        if rel_path == "scripts/check_release_ready.py":
            continue
        if not path.is_file() or path.suffix.lower() in {".pdf", ".cls", ".sty"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sanitized_text = text
        for snippet in ALLOWED_TEXT_SNIPPETS:
            sanitized_text = sanitized_text.replace(snippet, "")
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern.search(sanitized_text):
                text_leaks.append(
                    {
                        "path": rel_path,
                        "pattern": pattern.pattern,
                    }
                )
                break

    manifest_path = root / "RELEASE_MANIFEST.json"
    manifest_ok = False
    if manifest_path.exists():
        try:
            json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_ok = True
        except json.JSONDecodeError:
            manifest_ok = False

    report = {
        "root": str(root),
        "missing_required_files": missing,
        "forbidden_files": forbidden,
        "unexpected_paths": unexpected_paths,
        "text_leaks": text_leaks,
        "manifest_json_valid": manifest_ok,
        "n_files": sum(1 for path in root.rglob("*") if path.is_file()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if missing or forbidden or unexpected_paths or text_leaks or not manifest_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
