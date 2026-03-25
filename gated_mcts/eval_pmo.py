from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gated_mcts.utils.pmo_metrics import build_buffer_from_history, compute_pmo_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history_csv", type=str, help="Path to history csv (call_idx,smiles,score)")
    parser.add_argument("--freq_log", type=int, default=100, help="AUC logging frequency")
    parser.add_argument("--max_oracle_calls", type=int, default=10000, help="Oracle call budget")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output json path")
    args = parser.parse_args()

    history_path = Path(args.history_csv)
    df = pd.read_csv(history_path)
    buffer = build_buffer_from_history(df)
    metrics = compute_pmo_metrics(
        buffer,
        freq_log=int(args.freq_log),
        max_oracle_calls=int(args.max_oracle_calls),
    )

    print(f"History file:\t{history_path}")
    print(f"Unique calls:\t{metrics['n_calls']}")
    print(f"Avg. Top-1:\t{metrics['top1']:.6f}")
    print(f"Avg. Top-10:\t{metrics['top10']:.6f}")
    print(f"Avg. Top-100:\t{metrics['top100']:.6f}")
    print(f"AUC Top-1:\t{metrics['auc_top1']:.6f}")
    print(f"AUC Top-10:\t{metrics['auc_top10']:.6f}")
    print(f"AUC Top-100:\t{metrics['auc_top100']:.6f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics json:\t{out_path}")


if __name__ == "__main__":
    main()
