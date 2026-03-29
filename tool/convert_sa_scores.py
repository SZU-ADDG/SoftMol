from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RAW_METRIC_KEYS = (
    "top1",
    "top10",
    "top100",
    "auc_top1",
    "auc_top10",
    "auc_top100",
    "best_rv",
)


def sa_raw_from_norm(score_norm: float) -> float:
    return 10.0 - 9.0 * float(score_norm)


def _convert_score_csv(src_path: Path) -> Path:
    dst_path = src_path.with_name(f"{src_path.stem}_raw_sa{src_path.suffix}")
    with src_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "score" not in fieldnames:
            raise ValueError(f"'score' column not found in {src_path}")
        if "sa_raw" not in fieldnames:
            fieldnames.append("sa_raw")

        rows = []
        for row in reader:
            score = float(row["score"])
            row["sa_raw"] = f"{sa_raw_from_norm(score):.12f}"
            rows.append(row)

    with dst_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return dst_path


def _convert_metrics_json(src_path: Path) -> Path:
    dst_path = src_path.with_name(f"{src_path.stem}_raw_sa{src_path.suffix}")
    with src_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["score_transform"] = "sa_raw = 10 - 9 * score_norm"
    payload["source_oracle_name"] = payload.get("oracle_name", "sa")
    payload["oracle_name"] = "sa_raw"

    for file_key in ("history_file", "top10_file", "top100_file", "topk_file"):
        file_name = payload.get(file_key)
        if isinstance(file_name, str) and file_name:
            src_name = Path(file_name)
            payload[file_key] = f"{src_name.stem}_raw_sa{src_name.suffix}"

    for key in RAW_METRIC_KEYS:
        value = payload.get(key)
        if value is not None:
            payload[f"{key}_raw_sa"] = sa_raw_from_norm(float(value))

    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return dst_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add raw SA values to SoftMol property-MCTS outputs generated with oracle_name=sa."
    )
    parser.add_argument("--result_dir", type=Path, required=True, help="Directory containing the result files")
    parser.add_argument("--prefix", type=str, required=True, help="Common result prefix, e.g. sa_seed42")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    prefix = args.prefix

    history_path = result_dir / f"{prefix}_history.csv"
    metrics_path = result_dir / f"{prefix}_metrics.json"
    topk_paths = sorted(
        path for path in result_dir.glob(f"{prefix}_top*.csv") if not path.stem.endswith("_raw_sa")
    )

    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    written_paths = [_convert_score_csv(history_path), _convert_metrics_json(metrics_path)]
    written_paths.extend(_convert_score_csv(path) for path in topk_paths)

    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
