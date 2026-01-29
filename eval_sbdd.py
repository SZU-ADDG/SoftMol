"""Evaluate generated molecules for SBBD task.
Pipeline: deduplicate -> QED > 0.5 & SA < 5.0 -> docking score passes target threshold
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from tdc import Oracle

RDLogger.DisableLog("rdApp.*")

MAX_ROWS = 3000
QED_THR = 0.5
SA_THR = 5.0
TOP_FRAC = 0.05

HIT_THR_BY_TARGET: dict[str, float] = {
    "parp1": 10.0,
    "fa7": 8.5,
    "5ht1b": 8.7845,
    "braf": 10.3,
    "jak2": 9.1,
}


def _eval_qed_sa(smiles: list[str]) -> tuple[list[float], list[float]]:
    """Compute QED and SA for deduplicated SMILES."""
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    qed_list = [float(oracle_qed(s)) if s else math.nan for s in smiles]
    sa_list = [float(oracle_sa(s)) if s else math.nan for s in smiles]
    return qed_list, sa_list


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="results/sbdd/softmol/main/parp1_seed44.csv",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="parp1",
        choices=["parp1", "fa7", "5ht1b", "braf", "jak2"],
    )
    args = parser.parse_args()

    target = args.target
    hit_thr = HIT_THR_BY_TARGET[target]

    path = Path(args.input)
    df = pd.read_csv(path).iloc[:MAX_ROWS].copy()
    print(f"Raw samples (first {MAX_ROWS} rows):\t{len(df)}")

    df = df.dropna(subset=["smi"])
    df["smi"] = df["smi"].astype(str).str.strip()
    df = df[df["smi"] != ""]
    df = df.drop_duplicates(subset=["smi"])
    df["rv"] = pd.to_numeric(df["rv"], errors="coerce")

    df["qed"], df["sa"] = _eval_qed_sa(df["smi"].tolist())
    n_total = len(df)
    print(f"Deduplicated samples:\t\t{n_total}")

    df_qs = df[(df["qed"] > QED_THR) & (df["sa"] < SA_THR)]
    print(f"After QED/SA filter:\t\t{len(df_qs)}/{n_total}\t({len(df_qs) / n_total:.2%})")

    if df_qs.empty:
        return

    df_hit = df_qs[df_qs["rv"] > hit_thr]
    n_hit = len(df_hit)
    print(
        f"Hit (QED>{QED_THR}, SA<{SA_THR}, rv>{hit_thr:.4g} @ {target}):\t{n_hit}/{n_total}\t({n_hit / n_total:.2%})"
    )

    if df_hit.empty:
        return

    df_hit = df_hit.sort_values(by="rv", ascending=False)
    rv_vals = df_hit["rv"].to_numpy()
    rv_mean = float(rv_vals.mean())
    rv_top1 = float(rv_vals[0])
    k_top5 = max(1, int(math.ceil(len(rv_vals) * TOP_FRAC)))
    rv_top5_mean = float(rv_vals[:k_top5].mean())

    print("=" * 50)
    print(f"rv: top5%: {rv_top5_mean:.6f}; mean: {rv_mean:.6f}; top1: {rv_top1:.6f}")

    top5 = df_hit.head(k_top5)
    print(f"top5%  QED:\t\t{float(top5['qed'].mean()):.4f}")
    print(f"top5%  SA:\t\t{float(top5['sa'].mean()):.4f}")

    top5_path = path.with_name(path.stem + "_top5.csv")
    top5.to_csv(top5_path, index=False)
    print(f"Saved top5% samples to:\t\t{top5_path}")


if __name__ == "__main__":
    main()
