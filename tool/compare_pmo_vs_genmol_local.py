#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _resolve_path(project_root: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (project_root / path)


def _status(delta: float) -> str:
    if delta > 0:
        return "win"
    if delta < 0:
        return "lose"
    return "tie"


def _validate_columns(df: pd.DataFrame, required: set[str], source_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SoftMol PMO metrics against local Genmol PMO metrics."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--softmol_input_dir", type=str, default="results/pmo/softmol_mcts_seed42_20260327")
    parser.add_argument("--softmol_metrics_csv", type=str, default=None)
    parser.add_argument(
        "--genmol_metrics_csv",
        type=str,
        default=None,
        help="Path to genmol pmo_metrics_seed*.csv (local run output).",
    )
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    default_softmol_metrics = Path(args.softmol_input_dir) / f"pmo_metrics_seed{int(args.seed)}.csv"
    softmol_metrics_csv = _resolve_path(
        project_root,
        args.softmol_metrics_csv if args.softmol_metrics_csv is not None else str(default_softmol_metrics),
    )
    genmol_metrics_csv = _resolve_path(
        project_root,
        args.genmol_metrics_csv
        if args.genmol_metrics_csv is not None
        else str(
            Path("../genmol/scripts/exps/pmo/main/genmol/results")
            / f"pmo_metrics_seed{int(args.seed)}.csv"
        ),
    )
    output_csv = _resolve_path(
        project_root,
        args.output_csv
        if args.output_csv is not None
        else str(Path(args.softmol_input_dir) / f"compare_seed{int(args.seed)}_softmol_vs_genmol_local.csv"),
    )
    output_md = _resolve_path(
        project_root,
        args.output_md
        if args.output_md is not None
        else str(Path(args.softmol_input_dir) / f"compare_seed{int(args.seed)}_softmol_vs_genmol_local.md"),
    )

    if not softmol_metrics_csv.exists():
        raise FileNotFoundError(f"SoftMol metrics CSV not found: {softmol_metrics_csv}")
    if not genmol_metrics_csv.exists():
        raise FileNotFoundError(f"Genmol metrics CSV not found: {genmol_metrics_csv}")

    soft_df = pd.read_csv(softmol_metrics_csv)
    gen_df = pd.read_csv(genmol_metrics_csv)

    _validate_columns(
        soft_df,
        {"oracle", "auc_top10", "final_top10"},
        f"SoftMol metrics ({softmol_metrics_csv})",
    )
    _validate_columns(
        gen_df,
        {"oracle", "auc_top10", "final_top10"},
        f"Genmol metrics ({genmol_metrics_csv})",
    )

    soft_df = soft_df.rename(
        columns={
            "auc_top10": "softmol_auc_top10",
            "final_top10": "softmol_final_top10",
        }
    )
    gen_df = gen_df.rename(
        columns={
            "auc_top10": "genmol_auc_top10",
            "final_top10": "genmol_final_top10",
        }
    )

    soft_cols = ["oracle", "softmol_auc_top10", "softmol_final_top10"]
    gen_cols = ["oracle", "genmol_auc_top10", "genmol_final_top10"]
    soft_task_df = soft_df[soft_cols].drop_duplicates(subset=["oracle"])
    gen_task_df = gen_df[gen_cols].drop_duplicates(subset=["oracle"])

    soft_oracles = set(soft_task_df["oracle"].tolist())
    gen_oracles = set(gen_task_df["oracle"].tolist())
    missing_in_softmol = sorted(gen_oracles - soft_oracles)
    missing_in_genmol = sorted(soft_oracles - gen_oracles)
    if missing_in_softmol or missing_in_genmol:
        raise ValueError(
            f"Oracle mismatch: missing_in_softmol={missing_in_softmol}, "
            f"missing_in_genmol={missing_in_genmol}"
        )

    merged = soft_task_df.merge(gen_task_df, on="oracle", how="inner")
    merged["delta_auc_top10"] = merged["softmol_auc_top10"] - merged["genmol_auc_top10"]
    merged["delta_final_top10"] = merged["softmol_final_top10"] - merged["genmol_final_top10"]
    merged["status_auc_top10"] = merged["delta_auc_top10"].apply(_status)
    merged["status_final_top10"] = merged["delta_final_top10"].apply(_status)
    merged = merged.sort_values("oracle").reset_index(drop=True)

    sum_row = {
        "oracle": "Sum",
        "softmol_auc_top10": float(merged["softmol_auc_top10"].sum()),
        "genmol_auc_top10": float(merged["genmol_auc_top10"].sum()),
        "delta_auc_top10": float(merged["delta_auc_top10"].sum()),
        "softmol_final_top10": float(merged["softmol_final_top10"].sum()),
        "genmol_final_top10": float(merged["genmol_final_top10"].sum()),
        "delta_final_top10": float(merged["delta_final_top10"].sum()),
        "status_auc_top10": "",
        "status_final_top10": "",
    }
    mean_row = {
        "oracle": "Mean",
        "softmol_auc_top10": float(merged["softmol_auc_top10"].mean()),
        "genmol_auc_top10": float(merged["genmol_auc_top10"].mean()),
        "delta_auc_top10": float(merged["delta_auc_top10"].mean()),
        "softmol_final_top10": float(merged["softmol_final_top10"].mean()),
        "genmol_final_top10": float(merged["genmol_final_top10"].mean()),
        "delta_final_top10": float(merged["delta_final_top10"].mean()),
        "status_auc_top10": "",
        "status_final_top10": "",
    }
    out_df = pd.concat([merged, pd.DataFrame([sum_row, mean_row])], ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    auc_win = int((merged["status_auc_top10"] == "win").sum())
    auc_lose = int((merged["status_auc_top10"] == "lose").sum())
    auc_tie = int((merged["status_auc_top10"] == "tie").sum())
    final_win = int((merged["status_final_top10"] == "win").sum())
    final_lose = int((merged["status_final_top10"] == "lose").sum())
    final_tie = int((merged["status_final_top10"] == "tie").sum())

    best_auc = merged.sort_values("delta_auc_top10", ascending=False).head(3)
    worst_auc = merged.sort_values("delta_auc_top10", ascending=True).head(3)
    best_final = merged.sort_values("delta_final_top10", ascending=False).head(3)
    worst_final = merged.sort_values("delta_final_top10", ascending=True).head(3)

    lines = [
        f"# SoftMol vs Genmol PMO Comparison (seed={int(args.seed)})",
        "",
        f"- softmol_metrics_csv: `{softmol_metrics_csv}`",
        f"- genmol_metrics_csv: `{genmol_metrics_csv}`",
        f"- output_csv: `{output_csv}`",
        "",
        "## Overall",
        f"- n_tasks: {len(merged)}",
        f"- sum_delta_auc_top10: {sum_row['delta_auc_top10']:+.6f}",
        f"- mean_delta_auc_top10: {mean_row['delta_auc_top10']:+.6f}",
        f"- sum_delta_final_top10: {sum_row['delta_final_top10']:+.6f}",
        f"- mean_delta_final_top10: {mean_row['delta_final_top10']:+.6f}",
        "",
        "## Win/Lose/Tie",
        f"- auc_top10: {auc_win}/{auc_lose}/{auc_tie}",
        f"- final_top10: {final_win}/{final_lose}/{final_tie}",
        "",
        "## Top AUC Gains",
    ]
    for _, row in best_auc.iterrows():
        lines.append(f"- {row['oracle']}: {float(row['delta_auc_top10']):+.6f}")
    lines.extend(["", "## Top AUC Losses"])
    for _, row in worst_auc.iterrows():
        lines.append(f"- {row['oracle']}: {float(row['delta_auc_top10']):+.6f}")
    lines.extend(["", "## Top FinalTop10 Gains"])
    for _, row in best_final.iterrows():
        lines.append(f"- {row['oracle']}: {float(row['delta_final_top10']):+.6f}")
    lines.extend(["", "## Top FinalTop10 Losses"])
    for _, row in worst_final.iterrows():
        lines.append(f"- {row['oracle']}: {float(row['delta_final_top10']):+.6f}")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote compare CSV: {output_csv}")
    print(f"Wrote compare MD:  {output_md}")
    print(
        "AUC win/lose/tie={}/{}/{} | FinalTop10 win/lose/tie={}/{}/{}".format(
            auc_win, auc_lose, auc_tie, final_win, final_lose, final_tie
        )
    )


if __name__ == "__main__":
    main()
