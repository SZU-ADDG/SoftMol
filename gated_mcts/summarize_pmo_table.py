from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gated_mcts.utils.pmo_metrics import build_buffer_from_history, compute_pmo_metrics
from gated_mcts.utils.pmo_oracle_adapter import PMO_ORACLES


def _parse_seeds(text: str) -> List[int]:
    parts = [x.strip() for x in text.split(",")]
    seeds: List[int] = []
    for p in parts:
        if not p:
            continue
        seeds.append(int(p))
    if not seeds:
        raise ValueError("No seeds parsed.")
    return seeds


def _load_auc_top10(base_dir: Path, oracle_name: str, seed: int, freq_log: int, max_oracle_calls: int) -> float | None:
    prefix = f"{oracle_name}_seed{seed}"
    metrics_path = base_dir / f"{prefix}_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data["auc_top10"])
        except Exception:
            pass

    history_path = base_dir / f"{prefix}_history.csv"
    if history_path.exists():
        try:
            df = pd.read_csv(history_path)
            buffer = build_buffer_from_history(df)
            m = compute_pmo_metrics(buffer, freq_log=freq_log, max_oracle_calls=max_oracle_calls)
            return float(m["auc_top10"])
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results/pmo/softmol_mcts", help="Directory of PMO outputs")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seed list")
    parser.add_argument("--freq_log", type=int, default=100, help="AUC logging frequency")
    parser.add_argument("--max_oracle_calls", type=int, default=10000, help="Oracle call budget")
    parser.add_argument("--output_csv", type=str, default=None, help="Output csv path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    seeds = _parse_seeds(args.seeds)
    out_csv = Path(args.output_csv) if args.output_csv else input_dir / "table3_softmol_mcts.csv"

    rows: List[Dict[str, float | str]] = []
    for oracle_name in PMO_ORACLES:
        row: Dict[str, float | str] = {"oracle": oracle_name}
        vals: List[float] = []
        for seed in seeds:
            auc = _load_auc_top10(
                base_dir=input_dir,
                oracle_name=oracle_name,
                seed=seed,
                freq_log=int(args.freq_log),
                max_oracle_calls=int(args.max_oracle_calls),
            )
            key = f"seed{seed}"
            row[key] = np.nan if auc is None else float(auc)
            if auc is not None:
                vals.append(float(auc))
        row["mean_auc_top10"] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    sum_row: Dict[str, float | str] = {"oracle": "Sum"}
    for seed in seeds:
        col = f"seed{seed}"
        sum_row[col] = float(df[col].sum(skipna=True)) if col in df.columns else np.nan
    sum_row["mean_auc_top10"] = float(df["mean_auc_top10"].sum(skipna=True))

    df_out = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    print(f"Saved table:\t{out_csv}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
