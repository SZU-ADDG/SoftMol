#!/usr/bin/env python3
"""Batch-evaluate search strategy CSVs with eval_sbdd-aligned metrics.
python tool/eval_search_strategy.py \
  --root results/search_strategy \
  --max_rows 180 \
  --output_dir results/search_strategy
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_sbdd import (  # noqa: E402
    HIT_THR_BY_TARGET,
    QED_THR,
    SA_NORM_THR,
    reward_qed,
    reward_sa,
)

STRATEGIES = ["denovo", "greedy", "beam", "mcts"]
TARGETS = ["parp1", "fa7", "5ht1b", "braf", "jak2"]
METRICS = [
    "n_input",
    "n_valid",
    "n_unique",
    "hit_ratio",
    "hit_mean_ds",
    "top5_ds",
]
SUMMARY_METRICS = ["hit_ratio", "hit_mean_ds", "top5_ds"]
METHOD_DISPLAY = {
    "denovo": "de novo",
    "greedy": "Greedy",
    "beam": "Beam",
    "mcts": "MCTS",
}


def _read_and_normalize(path: Path, strategy: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path).iloc[:max_rows].copy()
    if strategy in {"mcts", "greedy", "beam"}:
        if "smi" not in df.columns or "rv" not in df.columns:
            raise ValueError(f"{path} missing required columns ['smi','rv'], got: {list(df.columns)}")
        out = pd.DataFrame({"smi": df["smi"].astype(str), "rv": pd.to_numeric(df["rv"], errors="coerce")})
        return out

    if strategy == "denovo":
        if "smiles" not in df.columns or "affinity" not in df.columns:
            raise ValueError(f"{path} missing required columns ['smiles','affinity'], got: {list(df.columns)}")
        affinity = pd.to_numeric(df["affinity"], errors="coerce")
        out = pd.DataFrame({"smi": df["smiles"].astype(str), "rv": -affinity})
        return out

    raise ValueError(f"Unsupported strategy: {strategy}")


def _evaluate_one(
    df_raw: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    n_input = int(len(df_raw))
    if n_input == 0:
        return {
            "n_input": 0,
            "n_valid": 0,
            "n_unique": 0,
            "hit_ratio": math.nan,
            "hit_mean_ds": math.nan,
            "top5_ds": math.nan,
        }

    hit_thr = float(HIT_THR_BY_TARGET[target])
    df = df_raw.copy()
    df["SMILES"] = df["smi"].astype(str)
    df["DOCKING"] = pd.to_numeric(df["rv"], errors="coerce")
    from rdkit import Chem  # local import to keep startup close to eval_sbdd style

    df["MOL"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["MOL"]).copy()
    n_valid = int(len(df))
    if n_valid == 0:
        return {
            "n_input": n_input,
            "n_valid": 0,
            "n_unique": 0,
            "hit_ratio": 0.0,
            "hit_mean_ds": math.nan,
            "top5_ds": math.nan,
        }

    df = df.drop_duplicates(subset=["SMILES"]).copy()
    n_unique = int(len(df))

    if "QED" not in df:
        df["QED"] = reward_qed(df["MOL"].tolist())
    if "SA" not in df:
        df["SA"] = reward_sa(df["MOL"].tolist())

    df = df[df["QED"] > QED_THR]
    df = df[df["SA"] > SA_NORM_THR]

    # Hit subset: QED/SA passed and rv > threshold.
    df_hit = df[df["DOCKING"] > hit_thr].copy()
    hit_ratio = float(len(df_hit) / n_input)
    if len(df_hit) > 0:
        # Mean DS computed strictly within hit subset.
        hit_mean_ds = float(df_hit["DOCKING"].mean())
        topk = max(1, int(math.ceil(len(df_hit) * 0.05)))
        top5_ds = float(df_hit.sort_values(by="DOCKING", ascending=False)["DOCKING"].iloc[:topk].mean())
    else:
        hit_mean_ds = math.nan
        top5_ds = math.nan

    return {
        "n_input": n_input,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "hit_ratio": hit_ratio,
        "hit_mean_ds": float(hit_mean_ds) if not pd.isna(hit_mean_ds) else math.nan,
        "top5_ds": float(top5_ds) if not pd.isna(top5_ds) else math.nan,
    }


def _format_metric(v: float, metric: str) -> str:
    if pd.isna(v):
        return "NA"
    if metric in {"n_input", "n_valid", "n_unique"}:
        return str(int(round(float(v))))
    return f"{float(v):.2f}"


def _build_summary_metric_table(
    runs_df: pd.DataFrame,
    metric: str,
    method_order: list[str],
    targets: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy in method_order:
        row: dict[str, object] = {"Method": METHOD_DISPLAY.get(strategy, strategy)}
        for t in targets:
            x = runs_df[(runs_df["strategy"] == strategy) & (runs_df["target"] == t)]
            row[t] = np.nan if x.empty else x.iloc[0][metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(
    runs_df: pd.DataFrame,
    strategies: list[str],
    targets: list[str],
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Search Strategy Evaluation Summary")
    lines.append("")
    lines.append("Hit metrics use only QED/SA filtering + rv threshold (no SIM filtering).")
    lines.append("")

    for metric in SUMMARY_METRICS:
        if metric == "hit_ratio":
            pretty_metric = "Hit ratio"
        elif metric == "hit_mean_ds":
            pretty_metric = "Hit mean DS (within Hit)"
        else:
            pretty_metric = "Top 5% DS (within Hit)"
        lines.append(f"## {pretty_metric}")
        lines.append("")
        tbl = _build_summary_metric_table(runs_df, metric, strategies, targets)
        header_cols = ["Method"] + targets
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
        for _, r in tbl.iterrows():
            vals = [_format_metric(r[t], metric) for t in targets]
            lines.append("| " + str(r["Method"]) + " | " + " | ".join(vals) + " |")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search_strategy results and summarize tables.")
    parser.add_argument("--root", type=str, default="results/search_strategy", help="Root directory of strategy CSVs")
    parser.add_argument("--max_rows", type=int, default=180, help="Use top-N rows per (strategy,target)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir (default: same as --root)")
    parser.add_argument("--sim_thr", type=float, default=0.4, help="Deprecated/unused (SIM filtering removed)")
    parser.add_argument("--zinc250k-csv", type=str, default=None, help="Deprecated/unused (SIM filtering removed)")
    parser.add_argument("--valid-idx-json", type=str, default=None, help="Deprecated/unused (SIM filtering removed)")
    parser.add_argument("--novel-cache", type=str, default=None, help="Deprecated/unused (SIM filtering removed)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    out_dir = root if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for strategy in args.strategies:
        for target in args.targets:
            path = root / strategy / f"{target}.csv"
            if not path.exists():
                records.append(
                    {
                        "strategy": strategy,
                        "target": target,
                        "input_csv": str(path),
                        "status": "missing",
                        **{k: math.nan for k in METRICS},
                    }
                )
                continue

            df_norm = _read_and_normalize(path=path, strategy=strategy, max_rows=int(args.max_rows))
            metrics = _evaluate_one(df_raw=df_norm, target=target)
            records.append(
                {
                    "strategy": strategy,
                    "target": target,
                    "input_csv": str(path),
                    "status": "ok",
                    **metrics,
                }
            )
            print(
                f"[ok] {strategy}/{target}: n_input={int(metrics['n_input'])}, "
                f"hit_ratio={metrics['hit_ratio']:.2f}, "
                f"hit_mean_ds={metrics['hit_mean_ds']:.2f}, "
                f"top5_ds={metrics['top5_ds']:.2f}"
            )

    runs_df = pd.DataFrame(records)
    runs_path = out_dir / "eval_runs.csv"
    runs_df.to_csv(runs_path, index=False, na_rep="NA")

    summary_rows: list[dict[str, object]] = []
    for metric in SUMMARY_METRICS:
        table = _build_summary_metric_table(runs_df, metric, list(args.strategies), list(args.targets))
        for t in args.targets:
            table[t] = table[t].apply(lambda v, m=metric: _format_metric(v, m))
        table.insert(0, "metric", metric)
        summary_rows.append(table)
    summary_df = pd.concat(summary_rows, axis=0, ignore_index=True)
    summary_path = out_dir / "eval_summary.csv"
    summary_df.to_csv(summary_path, index=False, na_rep="NA")

    summary_md_path = out_dir / "eval_summary.md"
    _write_markdown(runs_df=runs_df, strategies=list(args.strategies), targets=list(args.targets), out_path=summary_md_path)

    print(f"Saved runs: {runs_path}")
    print(f"Saved summary csv: {summary_path}")
    print(f"Saved summary md: {summary_md_path}")


if __name__ == "__main__":
    main()
