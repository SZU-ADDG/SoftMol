#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 merge_mcts.py 输出计算 QED/SA hit 相关评估指标。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from tdc import Oracle

RDLogger.DisableLog("rdApp.*")

HIT_THR_BY_TARGET: dict[str, float] = {
    "parp1": 10.0,
    "fa7": 8.5,
    "5ht1b": 8.7845,
    "braf": 10.3,
    "jak2": 9.1,
}


def _eval_qed_sa(smiles: list[str]) -> tuple[list[float], list[float]]:
    oracle_qed = Oracle("qed")
    oracle_sa = Oracle("sa")
    qed_list = [float(oracle_qed(s)) if s else math.nan for s in smiles]
    sa_list = [float(oracle_sa(s)) if s else math.nan for s in smiles]
    return qed_list, sa_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="合并后的 CSV 路径")
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True,
        choices=["parp1", "fa7", "5ht1b", "braf", "jak2"],
    )
    args = parser.parse_args()

    target = args.target.lower()
    hit_thr = HIT_THR_BY_TARGET.get(target)

    path = Path(args.input)
    df = pd.read_csv(path).iloc[:3000].copy()
    n_total_raw = len(df)
    print(f"原始样本数（前 3000）:\t{n_total_raw}")

    df = df.dropna(subset=["smi"])
    df["smi"] = df["smi"].astype(str).str.strip()
    df = df[df["smi"] != ""]
    df = df.drop_duplicates(subset=["smi"])
    df["rv"] = pd.to_numeric(df["rv"], errors="coerce")

    smiles = df["smi"].tolist()
    qed_list, sa_list = _eval_qed_sa(smiles)
    df["qed"] = qed_list
    df["sa"] = sa_list

    n_total = len(df)
    print(f"去重后样本数:\t\t{n_total}")

    df_qs = df[(df["qed"] > 0.5) & (df["sa"] < 5.0)]
    n_qs = len(df_qs)
    qs_ratio = n_qs / n_total if n_total else 0.0
    print(f"QED/SA 过滤后样本:\t{n_qs}/{n_total}\t({qs_ratio:.2%})")

    if n_qs == 0 or hit_thr is None:
        return

    df_qs = df_qs.dropna(subset=["rv"])
    df_hit = df_qs[df_qs["rv"] > hit_thr]
    n_hit = len(df_hit)
    hit_ratio = n_hit / n_total if n_total else 0.0
    print(
        f"hit (QED>0.5 且 SA<5.0 且 rv>{hit_thr:.4g} @ {target}):\t"
        f"{n_hit}/{n_total}\t({hit_ratio:.2%})"
    )

    if n_hit == 0:
        return

    df_hit = df_hit.sort_values(by="rv", ascending=False)
    rv_vals = df_hit["rv"].to_numpy()
    rv_mean = float(rv_vals.mean())
    rv_top1 = float(rv_vals[0])
    k_top5 = max(1, int(math.ceil(len(rv_vals) * 0.05)))
    rv_top5_mean = float(rv_vals[:k_top5].mean())

    print("=" * 50)
    print(f"rv: top5%: {rv_top5_mean:.6f}; mean: {rv_mean:.6f}; top1: {rv_top1:.6f}")

    top5 = df_hit.iloc[:k_top5]
    print(f"top5%  QED:\t\t{float(top5['qed'].mean()):.4f}")
    print(f"top5%  SA:\t\t{float(top5['sa'].mean()):.4f}")

    top5_path = path.with_name(path.stem + "_top5.csv")
    top5.sort_values(by="rv", ascending=False).to_csv(top5_path, index=False)
    print(f"保存 top5% 样本到:\t{top5_path}")


if __name__ == "__main__":
    main()
