#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并搜索输出 CSV（支持 mcts / greedy / beam）。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_PREFIXES = ("mcts_job", "greedy_job", "beam_job")


def _parse_prefix_arg(prefix_arg: str) -> list[str]:
    prefix_arg = (prefix_arg or "").strip()
    if not prefix_arg or prefix_arg.lower() == "auto":
        return []
    return [p.strip() for p in prefix_arg.split(",") if p.strip()]


def _find_csv_files(
    directory: Path,
    exclude: Sequence[Path],
    *,
    prefixes: Sequence[str],
) -> list[Path]:
    exclude_set = {p.resolve() for p in exclude}
    csv_files: list[Path] = []
    for entry in sorted(directory.iterdir()):
        if not (entry.is_file() and entry.suffix.lower() == ".csv"):
            continue
        if entry.resolve() in exclude_set:
            continue
        if prefixes and not any(entry.name.startswith(p) for p in prefixes):
            continue
        csv_files.append(entry)
    return csv_files


def _detect_prefixes(directory: Path, exclude: Sequence[Path]) -> list[str]:
    detected: list[str] = []
    for p in DEFAULT_PREFIXES:
        files = _find_csv_files(directory, exclude, prefixes=[p])
        if files:
            detected.append(p)
    return detected


def _merge_rows(csv_paths: Sequence[Path]) -> tuple[list[dict[str, str]], Sequence[str]]:
    aggregated_rows: list[dict[str, str]] = []
    header = None
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                continue
            if header is None:
                header = reader.fieldnames
            for row in reader:
                aggregated_rows.append({k: row.get(k, "") for k in header})
    if header is None:
        raise ValueError("没有找到任何有效 CSV。")
    return aggregated_rows, header


def _write_output(rows: Iterable[dict[str, str]], header: Sequence[str], output_path: Path) -> None:
    fieldnames = list(header)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 MCTS/Greedy/Beam CSV，不做排序和 QED/SA 计算。")
    parser.add_argument("-d", "--dir", type=Path, default=Path("."), help="扫描目录")
    parser.add_argument("-o", "--output", type=Path, help="输出路径，默认 <dir>/eval_merged.csv")
    parser.add_argument(
        "--prefix",
        type=str,
        default="auto",
        help="CSV 文件名前缀。可选 auto 或逗号分隔前缀（例如 mcts_job,greedy_job）",
    )
    args = parser.parse_args()

    directory = args.dir.resolve()
    if not directory.is_dir():
        parser.error(f"目录不存在：{directory}")

    output_path = args.output.resolve() if args.output else (directory / "eval_merged.csv").resolve()
    explicit_prefixes = _parse_prefix_arg(args.prefix)
    if explicit_prefixes:
        prefixes = explicit_prefixes
    else:
        prefixes = _detect_prefixes(directory, exclude=[output_path])

    if not prefixes:
        parser.error(
            f"未找到可合并 CSV。目录: {directory}；支持前缀: {', '.join(DEFAULT_PREFIXES)}；"
            "可用 --prefix 指定。"
        )

    csv_files = _find_csv_files(directory, exclude=[output_path], prefixes=prefixes)
    if not csv_files:
        parser.error(f"在目录 {directory} 中未找到前缀为 {prefixes} 的 CSV。")

    rows, header = _merge_rows(csv_files)
    print(f"前缀: {', '.join(prefixes)}")
    print(f"合并: {len(csv_files)} 个文件，共 {len(rows)} 行")
    _write_output(rows, header, output_path)
    print(f"写出: {output_path}")


if __name__ == "__main__":
    main()
