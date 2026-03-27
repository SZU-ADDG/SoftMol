#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gated_mcts.utils.pmo_metrics import build_buffer_from_history, compute_pmo_metrics
from gated_mcts.utils.pmo_oracle_adapter import PMO_ORACLES


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_oracles(raw: str | None) -> list[str]:
    if raw is None:
        return list(PMO_ORACLES)
    parsed = _parse_csv_list(raw)
    invalid = [item for item in parsed if item not in PMO_ORACLES]
    if invalid:
        raise ValueError(
            f"Unknown oracle(s): {invalid}. Supported oracles: {', '.join(PMO_ORACLES)}"
        )
    return parsed


def _compute_from_history(
    history_path: Path, max_oracle_calls: int, freq_log: int
) -> tuple[dict, int, int]:
    df = pd.read_csv(history_path)
    buffer_all = build_buffer_from_history(df)
    n_total = int(len(buffer_all))
    buffer_budget = {
        smiles: value for smiles, value in buffer_all.items() if int(value[1]) <= int(max_oracle_calls)
    }
    n_budget = int(len(buffer_budget))
    if n_budget == 0:
        raise ValueError(f"No molecule with call_idx <= {max_oracle_calls}: {history_path}")
    metrics_budget = compute_pmo_metrics(
        buffer_budget,
        freq_log=int(freq_log),
        max_oracle_calls=int(max_oracle_calls),
    )
    return metrics_budget, n_total, n_budget


def _compute_from_metrics(metrics_path: Path) -> tuple[dict, int, int]:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    n_calls = int(data.get("n_calls", 0))
    metrics_budget = {
        "auc_top10": float(data.get("auc_top10", 0.0)),
        "top10": float(data.get("top10", 0.0)),
    }
    return metrics_budget, n_calls, n_calls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SoftMol PMO outputs with Genmol-aligned schema."
    )
    parser.add_argument("--input_dir", type=str, default="results/pmo/softmol_mcts_seed42_20260327")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracles", type=str, default=None)
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--output_metrics_csv", type=str, default=None)
    parser.add_argument("--output_summary_csv", type=str, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    args = parser.parse_args()

    input_dir = _resolve_path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_metrics_csv = (
        _resolve_path(args.output_metrics_csv)
        if args.output_metrics_csv is not None
        else input_dir / f"pmo_metrics_seed{int(args.seed)}.csv"
    )
    output_summary_csv = (
        _resolve_path(args.output_summary_csv)
        if args.output_summary_csv is not None
        else input_dir / f"pmo_summary_seed{int(args.seed)}.csv"
    )

    rows: list[dict] = []
    missing: list[str] = []
    oracles = _parse_oracles(args.oracles)

    for oracle in oracles:
        prefix = f"{oracle}_seed{int(args.seed)}"
        metrics_path = input_dir / f"{prefix}_metrics.json"
        history_path = input_dir / f"{prefix}_history.csv"
        top10_path = input_dir / f"{prefix}_top10.csv"
        top100_path = input_dir / f"{prefix}_top100.csv"

        source = None
        if history_path.exists():
            metrics_budget, n_total, n_budget = _compute_from_history(
                history_path=history_path,
                max_oracle_calls=int(args.max_oracle_calls),
                freq_log=int(args.freq_log),
            )
            source = "history"
        elif metrics_path.exists():
            metrics_budget, n_total, n_budget = _compute_from_metrics(metrics_path)
            source = "metrics_json"
        else:
            missing.append(prefix)
            continue

        rows.append(
            {
                "oracle": oracle,
                "seed": int(args.seed),
                "auc_top10": float(metrics_budget["auc_top10"]),
                "final_top10": float(metrics_budget["top10"]),
                "n_molecules_total": int(n_total),
                "n_molecules_budget": int(n_budget),
                # Keep these names for schema-compatibility with genmol eval_batch.py.
                "yaml_file": str(metrics_path) if metrics_path.exists() else "",
                "csv_file": str(history_path) if history_path.exists() else "",
                "metrics_file": str(metrics_path) if metrics_path.exists() else "",
                "history_file": str(history_path) if history_path.exists() else "",
                "top10_file": str(top10_path) if top10_path.exists() else "",
                "top100_file": str(top100_path) if top100_path.exists() else "",
                "source": source,
            }
        )

    if missing and not args.skip_missing:
        raise FileNotFoundError(
            "Missing PMO outputs for:\n" + "\n".join(missing) + "\nUse --skip_missing to continue."
        )
    if not rows:
        raise RuntimeError("No valid PMO result found.")

    rows.sort(key=lambda item: item["oracle"])

    output_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "oracle",
                "seed",
                "auc_top10",
                "final_top10",
                "n_molecules_total",
                "n_molecules_budget",
                "yaml_file",
                "csv_file",
                "metrics_file",
                "history_file",
                "top10_file",
                "top100_file",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    sum_auc_top10 = float(sum(item["auc_top10"] for item in rows))
    sum_final_top10 = float(sum(item["final_top10"] for item in rows))
    n_tasks = int(len(rows))
    summary_row = {
        "seed": int(args.seed),
        "n_tasks": n_tasks,
        "sum_auc_top10": sum_auc_top10,
        "sum_final_top10": sum_final_top10,
        "mean_auc_top10": float(sum_auc_top10 / n_tasks),
        "mean_final_top10": float(sum_final_top10 / n_tasks),
        "max_oracle_calls": int(args.max_oracle_calls),
        "freq_log": int(args.freq_log),
        "missing_tasks": int(len(missing)),
    }

    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Wrote per-task metrics to: {output_metrics_csv}")
    print(f"Wrote summary metrics to: {output_summary_csv}")
    print(
        f"n_tasks={n_tasks} | "
        f"sum_auc_top10={sum_auc_top10:.6f} | "
        f"sum_final_top10={sum_final_top10:.6f}"
    )
    if missing:
        print(f"Skipped {len(missing)} missing task(s): {missing}")


if __name__ == "__main__":
    main()
