#!/usr/bin/env python3
"""Batch-run eval_sbdd.py and summarize mean±std across 3 seeds.

Outputs:
- results/sbdd/softmol/batch_eval_runs.csv
- results/sbdd/softmol/batch_eval_summary.csv
- results/sbdd/softmol/batch_eval_summary.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["parp1", "fa7", "5ht1b", "braf", "jak2"]
SEEDS = [42, 43, 44]

METHODS = {
    "main": {
        "method_name": "SoftMol",
        "rel_dir": "results/sbdd/softmol/main",
    },
    "unconstrained": {
        "method_name": "SoftMol (Unconstrained)",
        "rel_dir": "results/sbdd/softmol/unconstrained",
    },
}

METRIC_PATTERNS = {
    "novelty_raw": re.compile(r"Novelty:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "circle": re.compile(r"#Circle:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "novel_hit_ratio_raw": re.compile(r"Novel hit ratio:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "novel_top5_ds": re.compile(r"Novel top 5% DS:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}

METRIC_SPECS = [
    ("novelty_raw", "Novelty (%)", 100.0),
    ("circle", "#Circle", 1.0),
    ("novel_hit_ratio_raw", "Novel hit ratio (%)", 100.0),
    ("novel_top5_ds", "Novel top 5% DS", 1.0),
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parse_metrics(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, p in METRIC_PATTERNS.items():
        m = p.search(stdout)
        if not m:
            raise ValueError(f"Failed to parse metric '{k}' from eval output.")
        out[k] = float(m.group(1))
    return out


def _build_runs(
    root: Path,
    eval_script: Path,
    output_dir: Path,
    method_keys: list[str],
) -> pd.DataFrame:
    runs_csv_path = output_dir / "batch_eval_runs.csv"
    logs_dir = output_dir / "batch_eval_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total_jobs = len(method_keys) * len(TARGETS) * len(SEEDS)
    job_idx = 0

    for mk in method_keys:
        method_name = METHODS[mk]["method_name"]
        method_dir = root / METHODS[mk]["rel_dir"]
        if not method_dir.exists():
            raise FileNotFoundError(f"Method directory not found: {method_dir}")

        for target in TARGETS:
            for seed in SEEDS:
                job_idx += 1
                input_csv = method_dir / f"{target}_seed{seed}.csv"
                if not input_csv.exists():
                    raise FileNotFoundError(f"Input file not found: {input_csv}")

                cmd = [
                    sys.executable,
                    str(eval_script),
                    "-i",
                    str(input_csv),
                    "-t",
                    target,
                ]
                print(f"[{job_idx}/{total_jobs}] Running: {method_name} | {target} | seed{seed}")
                proc = subprocess.run(
                    cmd,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                )

                log_file = logs_dir / f"{_slug(method_name)}__{target}_seed{seed}.log"
                with log_file.open("w", encoding="utf-8") as f:
                    f.write("CMD: " + " ".join(cmd) + "\n")
                    f.write(f"RETURNCODE: {proc.returncode}\n")
                    f.write("\n[STDOUT]\n")
                    f.write(proc.stdout)
                    f.write("\n[STDERR]\n")
                    f.write(proc.stderr)

                if proc.returncode != 0:
                    if rows:
                        pd.DataFrame(rows).to_csv(runs_csv_path, index=False)
                    raise RuntimeError(
                        f"eval failed for {input_csv}\n"
                        f"See log: {log_file}\n"
                        f"stderr:\n{proc.stderr}"
                    )

                metrics = _parse_metrics(proc.stdout)
                row = {
                    "method": method_name,
                    "target": target,
                    "seed": seed,
                    "input_csv": str(input_csv),
                    "log_path": str(log_file),
                    **metrics,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(runs_csv_path, index=False)

    return pd.DataFrame(rows)


def _build_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for method_name in [METHODS["main"]["method_name"], METHODS["unconstrained"]["method_name"]]:
        df_m = runs_df[runs_df["method"] == method_name]
        for target in TARGETS:
            df_t = df_m[df_m["target"] == target]
            for raw_col, metric_name, scale in METRIC_SPECS:
                vals = df_t[raw_col].astype(float).to_numpy()
                if vals.size == 0:
                    mean_raw = np.nan
                    std_raw = np.nan
                    n_runs = 0
                else:
                    mean_raw = float(vals.mean())
                    std_raw = float(vals.std(ddof=0))
                    n_runs = int(vals.size)
                mean = mean_raw * scale
                std = std_raw * scale
                records.append(
                    {
                        "method": method_name,
                        "target": target,
                        "metric": metric_name,
                        "mean": mean,
                        "std": std,
                        "n_runs": n_runs,
                        "display": f"{mean:.3f} ± {std:.3f}",
                        "mean_raw": mean_raw,
                        "std_raw": std_raw,
                        "scale": scale,
                    }
                )

    summary_df = pd.DataFrame(records)
    metric_order = [m[1] for m in METRIC_SPECS]
    summary_df["method"] = pd.Categorical(
        summary_df["method"],
        categories=[METHODS["main"]["method_name"], METHODS["unconstrained"]["method_name"]],
        ordered=True,
    )
    summary_df["target"] = pd.Categorical(summary_df["target"], categories=TARGETS, ordered=True)
    summary_df["metric"] = pd.Categorical(summary_df["metric"], categories=metric_order, ordered=True)
    summary_df = summary_df.sort_values(["metric", "method", "target"]).reset_index(drop=True)
    return summary_df


def _write_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# SoftMol SBBD Batch Evaluation Summary")
    lines.append("")
    lines.append("Statistics are mean ± std across 3 seeds (42/43/44).")
    lines.append("")

    for _, metric_name, _ in METRIC_SPECS:
        lines.append(f"## {metric_name}")
        lines.append("")
        lines.append("| Method | parp1 | fa7 | 5ht1b | braf | jak2 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for method_name in [METHODS["main"]["method_name"], METHODS["unconstrained"]["method_name"]]:
            vals = []
            for t in TARGETS:
                x = summary_df[
                    (summary_df["metric"] == metric_name)
                    & (summary_df["method"] == method_name)
                    & (summary_df["target"] == t)
                ]
                if x.empty:
                    vals.append("NA")
                else:
                    vals.append(str(x.iloc[0]["display"]))
            lines.append("| " + method_name + " | " + " | ".join(vals) + " |")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch run eval_sbdd.py and summarize metrics by method/target."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="SoftMol repo root",
    )
    parser.add_argument(
        "--eval-script",
        type=str,
        default="eval_sbdd.py",
        help="Path to eval_sbdd.py (relative to --root or absolute)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for batch csv/md (default: <root>/results/sbdd/softmol)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["main", "unconstrained"],
        default=["main", "unconstrained"],
        help="Which method groups to run",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    eval_script = Path(args.eval_script)
    if not eval_script.is_absolute():
        eval_script = (root / eval_script).resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script}")

    output_dir = (
        (root / "results" / "sbdd" / "softmol").resolve()
        if args.output_dir is None
        else Path(args.output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = _build_runs(
        root=root,
        eval_script=eval_script,
        output_dir=output_dir,
        method_keys=args.methods,
    )

    runs_csv = output_dir / "batch_eval_runs.csv"
    summary_csv = output_dir / "batch_eval_summary.csv"
    summary_md = output_dir / "batch_eval_summary.md"

    runs_df.to_csv(runs_csv, index=False)
    summary_df = _build_summary(runs_df)
    summary_df.to_csv(summary_csv, index=False)
    _write_markdown(summary_df, summary_md)

    print("Batch evaluation done.")
    print(f"Runs CSV:     {runs_csv}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary MD:   {summary_md}")


if __name__ == "__main__":
    main()
