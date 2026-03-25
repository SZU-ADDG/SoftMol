from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def top_auc(
    buffer: Dict[str, Tuple[float, int]],
    top_n: int,
    finish: bool,
    freq_log: int,
    max_oracle_calls: int,
) -> float:
    """GenMol-compatible AUC implementation for PMO."""
    area = 0.0
    prev = 0.0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = float(np.mean([item[1][0] for item in temp_result])) if temp_result else 0.0
        area += freq_log * (top_n_now + prev) / 2.0
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = float(np.mean([item[1][0] for item in temp_result])) if temp_result else 0.0
    area += (len(buffer) - called) * (top_n_now + prev) / 2.0
    if finish and len(buffer) < max_oracle_calls:
        area += (max_oracle_calls - len(buffer)) * top_n_now
    return float(area / max_oracle_calls)


def build_buffer_from_history(df: pd.DataFrame) -> Dict[str, Tuple[float, int]]:
    """Build canonical PMO buffer from history rows.

    Expected columns:
    - smiles
    - score
    - optional call_idx
    """
    work = df.copy()
    if "call_idx" not in work.columns:
        work["call_idx"] = np.arange(1, len(work) + 1, dtype=int)

    work = work.dropna(subset=["smiles"]).copy()
    work["smiles"] = work["smiles"].astype(str).str.strip()
    work = work[work["smiles"] != ""]
    work["score"] = pd.to_numeric(work["score"], errors="coerce").fillna(0.0).astype(float)
    work["call_idx"] = pd.to_numeric(work["call_idx"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["call_idx"]).copy()
    work["call_idx"] = work["call_idx"].astype(int)
    work = work.sort_values("call_idx")

    # Keep first-seen index for each smiles; allow later rows to improve score if repeated.
    buffer: Dict[str, Tuple[float, int]] = {}
    for _, row in work.iterrows():
        smi = str(row["smiles"])
        score = float(row["score"])
        idx = int(row["call_idx"])
        if smi not in buffer:
            buffer[smi] = (score, idx)
        else:
            old_score, old_idx = buffer[smi]
            buffer[smi] = (max(old_score, score), old_idx)
    return buffer


def compute_pmo_metrics(
    buffer: Dict[str, Tuple[float, int]],
    *,
    freq_log: int = 100,
    max_oracle_calls: int = 10000,
) -> Dict[str, float]:
    if not buffer:
        return {
            "n_calls": 0,
            "top1": 0.0,
            "top10": 0.0,
            "top100": 0.0,
            "auc_top1": 0.0,
            "auc_top10": 0.0,
            "auc_top100": 0.0,
        }

    scores = np.array(sorted([v[0] for v in buffer.values()], reverse=True), dtype=float)
    top1 = float(scores[0]) if len(scores) >= 1 else 0.0
    top10 = float(scores[:10].mean()) if len(scores) >= 1 else 0.0
    top100 = float(scores[:100].mean()) if len(scores) >= 1 else 0.0

    return {
        "n_calls": int(len(buffer)),
        "top1": top1,
        "top10": top10,
        "top100": top100,
        "auc_top1": top_auc(buffer, 1, True, int(freq_log), int(max_oracle_calls)),
        "auc_top10": top_auc(buffer, 10, True, int(freq_log), int(max_oracle_calls)),
        "auc_top100": top_auc(buffer, 100, True, int(freq_log), int(max_oracle_calls)),
    }
