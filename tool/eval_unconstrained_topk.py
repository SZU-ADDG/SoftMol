#!/usr/bin/env python3
"""Evaluate unconstrained SBDD CSV files with RV/QED/SA top-k metrics.

Default inputs:
- results/sbdd/softmol/additional/unconstrained/1UWH.csv
- results/sbdd/softmol/additional/unconstrained/6GL8.csv

Top-k definition:
- rv: larger is better (descending)
- qed: larger is better (descending)
- sa: smaller is better (ascending)
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Row:
    smi: str
    rv: float
    qed: float
    sa: float


def _pick_column(columns: list[str], candidates: tuple[str, ...], name: str) -> str:
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"Missing {name} column, candidates={candidates}, got={columns}")


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        num = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _topk_stats(values: list[float], *, reverse: bool, ks: tuple[int, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not values:
        for k in ks:
            out[f"top{k}"] = float("nan")
            out[f"top{k}_mean"] = float("nan")
            out[f"k_top{k}"] = 0
        return out

    sorted_vals = sorted(values, reverse=reverse)
    for k in ks:
        kk = min(k, len(sorted_vals))
        head = sorted_vals[:kk]
        out[f"k_top{k}"] = kk
        out[f"top{k}"] = head[0] if head else float("nan")
        out[f"top{k}_mean"] = _safe_mean(head)
    return out


def _load_rows(input_csv: Path) -> tuple[list[Row], dict[str, str], dict[str, int]]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {input_csv}")
        columns = list(reader.fieldnames)

        smi_col = _pick_column(columns, ("smi", "smiles", "SMILES"), "SMILES")
        rv_col = _pick_column(columns, ("rv", "docking_score", "total", "score", "ds"), "RV/docking")
        has_qed = "qed" in columns
        has_sa = "sa" in columns

        q_oracle: Callable[[str], float] | None = None
        s_oracle: Callable[[str], float] | None = None
        if not (has_qed and has_sa):
            try:
                from tdc import Oracle  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Input CSV has no qed/sa columns, and tdc is unavailable to compute them. "
                    "Please install `tdc` (and its dependencies) or provide qed/sa columns."
                ) from e
            q_oracle = Oracle("qed")
            s_oracle = Oracle("sa")

        n_raw = 0
        n_clean = 0
        rows: list[Row] = []
        for row in reader:
            n_raw += 1

            smi = (row.get(smi_col) or "").strip()
            rv = _to_float(row.get(rv_col))
            if not smi or rv is None:
                continue

            if has_qed and has_sa:
                qed = _to_float(row.get("qed"))
                sa = _to_float(row.get("sa"))
            else:
                assert q_oracle is not None and s_oracle is not None
                try:
                    qed = float(q_oracle(smi))
                    sa = float(s_oracle(smi))
                except Exception:
                    qed = None
                    sa = None

            if qed is None or sa is None:
                continue

            n_clean += 1
            rows.append(Row(smi=smi, rv=rv, qed=qed, sa=sa))

    col_info = {
        "smi_col": smi_col,
        "rv_col": rv_col,
        "qed_col": "qed" if has_qed else "computed_by_tdc",
        "sa_col": "sa" if has_sa else "computed_by_tdc",
    }
    count_info = {"n_raw": n_raw, "n_clean": n_clean}
    return rows, col_info, count_info


def _dedup_keep_best_rv(rows: list[Row]) -> list[Row]:
    best: dict[str, Row] = {}
    for r in rows:
        old = best.get(r.smi)
        if old is None or r.rv > old.rv:
            best[r.smi] = r
    return list(best.values())


def _fmt(x: float) -> str:
    if x is None or math.isnan(x):
        return "nan"
    return f"{x:.6f}"


def _default_inputs(project_root: Path) -> list[Path]:
    base = project_root / "results" / "sbdd" / "softmol" / "additional" / "unconstrained"
    return [base / "1UWH.csv", base / "6GL8.csv"]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        default=[str(p) for p in _default_inputs(project_root)],
        help="Input CSV files",
    )
    parser.add_argument(
        "-o",
        "--output-summary",
        type=str,
        default=str(
            project_root
            / "results"
            / "sbdd"
            / "softmol"
            / "additional"
            / "unconstrained"
            / "topk_metrics_summary.csv"
        ),
        help="Output summary CSV path",
    )
    args = parser.parse_args()

    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    output_summary = Path(args.output_summary).expanduser().resolve()

    summary_rows: list[dict[str, str | int | float]] = []

    for input_csv in input_paths:
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        rows, col_info, count_info = _load_rows(input_csv)
        dedup_rows = _dedup_keep_best_rv(rows)

        rv_values = [r.rv for r in dedup_rows]
        qed_values = [r.qed for r in dedup_rows]
        sa_values = [r.sa for r in dedup_rows]

        rv_stats = _topk_stats(rv_values, reverse=True, ks=(1, 10, 100))
        qed_stats = _topk_stats(qed_values, reverse=True, ks=(1, 10, 100))
        sa_stats = _topk_stats(sa_values, reverse=False, ks=(1, 10, 100))

        n_dedup = len(dedup_rows)
        row: dict[str, str | int | float] = {
            "input_csv": str(input_csv),
            "n_raw": count_info["n_raw"],
            "n_clean": count_info["n_clean"],
            "n_dedup": n_dedup,
            "smi_col": col_info["smi_col"],
            "rv_col": col_info["rv_col"],
            "qed_source": col_info["qed_col"],
            "sa_source": col_info["sa_col"],
            "rv_top1": rv_stats["top1"],
            "rv_top10_mean": rv_stats["top10_mean"],
            "rv_top100_mean": rv_stats["top100_mean"],
            "qed_top1": qed_stats["top1"],
            "qed_top10_mean": qed_stats["top10_mean"],
            "qed_top100_mean": qed_stats["top100_mean"],
            "sa_top1": sa_stats["top1"],
            "sa_top10_mean": sa_stats["top10_mean"],
            "sa_top100_mean": sa_stats["top100_mean"],
        }
        summary_rows.append(row)

        print(f"Input CSV: {input_csv}")
        print(
            "Detected columns: "
            f"SMILES={col_info['smi_col']}, RV={col_info['rv_col']}, "
            f"QED={col_info['qed_col']}, SA={col_info['sa_col']}"
        )
        print(
            f"Rows raw/clean/dedup: {count_info['n_raw']}/"
            f"{count_info['n_clean']}/{n_dedup}"
        )
        print(
            f"RV   top1={_fmt(rv_stats['top1'])}, "
            f"top10_mean={_fmt(rv_stats['top10_mean'])}, "
            f"top100_mean={_fmt(rv_stats['top100_mean'])}"
        )
        print(
            f"QED  top1={_fmt(qed_stats['top1'])}, "
            f"top10_mean={_fmt(qed_stats['top10_mean'])}, "
            f"top100_mean={_fmt(qed_stats['top100_mean'])}"
        )
        print(
            f"SA   top1={_fmt(sa_stats['top1'])}, "
            f"top10_mean={_fmt(sa_stats['top10_mean'])}, "
            f"top100_mean={_fmt(sa_stats['top100_mean'])}"
        )
        print("-" * 72)

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "input_csv",
        "n_raw",
        "n_clean",
        "n_dedup",
        "smi_col",
        "rv_col",
        "qed_source",
        "sa_source",
        "rv_top1",
        "rv_top10_mean",
        "rv_top100_mean",
        "qed_top1",
        "qed_top10_mean",
        "qed_top100_mean",
        "sa_top1",
        "sa_top10_mean",
        "sa_top100_mean",
    ]
    with output_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary CSV: {output_summary}")


if __name__ == "__main__":
    main()
