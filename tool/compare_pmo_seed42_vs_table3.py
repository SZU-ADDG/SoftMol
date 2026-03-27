#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _resolve_path(project_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (project_root / p)


def _load_sum_value(df: pd.DataFrame, col: str, default_sum: float | None = None) -> float:
    row = df[df["oracle"] == "Sum"]
    if len(row) > 0:
        return float(row.iloc[0][col])
    if default_sum is None:
        raise ValueError(f"Missing Sum row for column: {col}")
    return float(default_sum)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PMO seed42 results and compare against Table-3 GenMol column.")
    parser.add_argument("--input_dir", type=str, default="results/pmo/softmol_mcts_seed42_20260327")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--reference_csv", type=str, default="tool/table3_genmol_auc_top10.csv")
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--skip_summarize", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_dir = _resolve_path(project_root, args.input_dir)
    seed_col = f"seed{int(args.seed)}"

    summary_csv = _resolve_path(
        project_root,
        args.summary_csv
        if args.summary_csv is not None
        else str(Path(args.input_dir) / "table3_softmol_mcts_seed42.csv"),
    )
    output_csv = _resolve_path(
        project_root,
        args.output_csv
        if args.output_csv is not None
        else str(Path(args.input_dir) / "compare_seed42_vs_table3_genmol.csv"),
    )
    output_md = _resolve_path(
        project_root,
        args.output_md
        if args.output_md is not None
        else str(Path(args.input_dir) / "compare_seed42_vs_table3_genmol.md"),
    )
    reference_csv = _resolve_path(project_root, args.reference_csv)

    if not input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {input_dir}")
    if not reference_csv.exists():
        raise SystemExit(f"Reference csv not found: {reference_csv}")

    if not args.skip_summarize:
        cmd = [
            sys.executable,
            str(project_root / "gated_mcts" / "summarize_pmo_table.py"),
            "--input_dir",
            str(input_dir),
            "--seeds",
            str(int(args.seed)),
            "--freq_log",
            str(int(args.freq_log)),
            "--max_oracle_calls",
            str(int(args.max_oracle_calls)),
            "--output_csv",
            str(summary_csv),
        ]
        print("[INFO] Running summarize command:")
        print("       " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    if not summary_csv.exists():
        raise SystemExit(f"Summary csv not found: {summary_csv}")

    summary_df = pd.read_csv(summary_csv)
    ref_df = pd.read_csv(reference_csv)

    required_summary_cols = {"oracle", seed_col}
    if not required_summary_cols.issubset(set(summary_df.columns)):
        raise SystemExit(f"Summary csv missing required columns: {required_summary_cols}")

    required_ref_cols = {"oracle", "genmol_auc_top10"}
    if not required_ref_cols.issubset(set(ref_df.columns)):
        raise SystemExit(f"Reference csv missing required columns: {required_ref_cols}")

    if len(summary_df) != 24:
        raise SystemExit(f"Summary csv should have 24 rows (23 + Sum), got {len(summary_df)}")

    if summary_df[seed_col].isna().any():
        bad_rows = summary_df[summary_df[seed_col].isna()]["oracle"].tolist()
        raise SystemExit(f"Summary csv has NaN in {seed_col}: {bad_rows}")

    summary_task_df = summary_df[summary_df["oracle"] != "Sum"].copy()
    ref_task_df = ref_df[ref_df["oracle"] != "Sum"].copy()

    if len(summary_task_df) != 23:
        raise SystemExit(f"Summary task rows should be 23, got {len(summary_task_df)}")
    if len(ref_task_df) != 23:
        raise SystemExit(f"Reference task rows should be 23, got {len(ref_task_df)}")

    summary_oracles = set(summary_task_df["oracle"].tolist())
    ref_oracles = set(ref_task_df["oracle"].tolist())
    missing_oracles = sorted(ref_oracles - summary_oracles)
    extra_oracles = sorted(summary_oracles - ref_oracles)
    if missing_oracles or extra_oracles:
        raise SystemExit(
            f"Oracle mismatch. missing={missing_oracles}, extra={extra_oracles}"
        )

    missing_metric_files: list[str] = []
    bad_n_calls: list[str] = []
    for oracle in sorted(ref_oracles):
        metrics_path = input_dir / f"{oracle}_seed{int(args.seed)}_metrics.json"
        if not metrics_path.exists():
            missing_metric_files.append(metrics_path.name)
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n_calls = int(data.get("n_calls", -1))
        if n_calls != int(args.max_oracle_calls):
            bad_n_calls.append(f"{metrics_path.name}:{n_calls}")

    if missing_metric_files:
        raise SystemExit(f"Missing metrics files ({len(missing_metric_files)}): {missing_metric_files}")
    if bad_n_calls:
        raise SystemExit(f"Metrics n_calls != {int(args.max_oracle_calls)}: {bad_n_calls}")

    merged = summary_task_df[["oracle", seed_col]].merge(ref_task_df, on="oracle", how="inner")
    merged["delta"] = merged[seed_col].astype(float) - merged["genmol_auc_top10"].astype(float)

    def _status(v: float) -> str:
        if v > 0:
            return "win"
        if v < 0:
            return "lose"
        return "tie"

    merged["status"] = merged["delta"].apply(_status)

    soft_sum = _load_sum_value(summary_df, seed_col, default_sum=float(merged[seed_col].sum()))
    ref_sum = _load_sum_value(ref_df, "genmol_auc_top10", default_sum=float(merged["genmol_auc_top10"].sum()))
    delta_sum = float(soft_sum - ref_sum)

    calc_sum = float(merged[seed_col].sum())
    if abs(calc_sum - soft_sum) > 1e-6:
        raise SystemExit(
            f"Sum consistency check failed: Sum({seed_col})={soft_sum}, computed={calc_sum}"
        )

    out_df = merged[["oracle", seed_col, "genmol_auc_top10", "delta", "status"]].copy()
    sum_row = pd.DataFrame(
        [
            {
                "oracle": "Sum",
                seed_col: soft_sum,
                "genmol_auc_top10": ref_sum,
                "delta": delta_sum,
                "status": "",
            }
        ]
    )
    out_df = pd.concat([out_df, sum_row], ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    win_count = int((merged["status"] == "win").sum())
    lose_count = int((merged["status"] == "lose").sum())
    tie_count = int((merged["status"] == "tie").sum())

    sorted_df = merged.sort_values("delta", ascending=False)
    top3_gain = sorted_df.head(3)[["oracle", "delta"]].values.tolist()
    top3_loss = sorted_df.tail(3)[["oracle", "delta"]].values.tolist()

    lines = [
        "# Seed42 PMO vs Table3 GenMol (Pre-comparison)",
        "",
        "> This is a single-seed pre-comparison (not 3-run mean).",
        "",
        f"- input_dir: `{input_dir}`",
        f"- summary_csv: `{summary_csv}`",
        f"- reference_csv: `{reference_csv}`",
        f"- compare_csv: `{output_csv}`",
        "",
        "## Checks",
        f"- metrics files: 23/23 present",
        f"- n_calls check: all == {int(args.max_oracle_calls)}",
        f"- summary rows: {len(summary_df)} (expect 24)",
        "",
        "## Overall",
        f"- win_count: {win_count}",
        f"- lose_count: {lose_count}",
        f"- tie_count: {tie_count}",
        f"- sum_softmol_seed42: {soft_sum:.6f}",
        f"- sum_genmol_table3: {ref_sum:.6f}",
        f"- delta_sum: {delta_sum:.6f}",
        "",
        "## Top Gains",
    ]

    for oracle, delta in top3_gain:
        lines.append(f"- {oracle}: {float(delta):+.6f}")

    lines += ["", "## Top Losses"]
    for oracle, delta in top3_loss:
        lines.append(f"- {oracle}: {float(delta):+.6f}")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] compare csv: {output_csv}")
    print(f"[OK] compare md:  {output_md}")
    print(f"[OK] win/lose/tie: {win_count}/{lose_count}/{tie_count}")
    print(f"[OK] delta_sum:    {delta_sum:+.6f}")


if __name__ == "__main__":
    main()
