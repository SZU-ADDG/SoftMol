#!/usr/bin/env python3
"""Randomly sample SMILES from a Hugging Face disk dataset split.

Example:
python tool/sample_train_10m.py \
  --dataset-path /share/home/tm866079609100000/a875465180/yqw_bd3lms/data/DrugLikeSMILSE-12B-427M-filterLen72 \
  --split train \
  --num-samples 10000000 \
  --seed 42 \
  --select-batch-size 200000 \
  --num-workers 1 \
  --output /share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol/tool/train_sample_10m.smi

"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk


DEFAULT_DATASET_PATH = (
    "/share/home/tm866079609100000/a875465180/yqw_bd3lms/data/DrugLikeSMILSE-12B-427M"
)
DEFAULT_OUTPUT = "./train_sample_10m.smi"
DEFAULT_SPLIT = "train"
DEFAULT_NUM_SAMPLES = 10_000_000
DEFAULT_SEED = 42
DEFAULT_SELECT_BATCH_SIZE = 200_000
DEFAULT_SMILES_COLUMN = "input"
DEFAULT_NUM_WORKERS = 1

_WORKER_SPLIT_DS = None
_WORKER_SMILES_COLUMN = None


class ProgressBar:
    """Lightweight terminal progress bar with no external dependency."""

    def __init__(self, total: int, desc: str, width: int = 32) -> None:
        self.total = max(1, int(total))
        self.desc = desc
        self.width = max(10, int(width))
        self.count = 0
        self._start = time.time()

    def _render(self, postfix: str = "") -> None:
        ratio = min(1.0, self.count / self.total)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = ratio * 100.0
        elapsed = time.time() - self._start
        msg = (
            f"\r[{self.desc}] {self.count}/{self.total} "
            f"|{bar}| {pct:6.2f}% elapsed={elapsed:6.1f}s"
        )
        if postfix:
            msg += f" {postfix}"
        print(msg, end="", flush=True)

    def update(self, n: int = 1, postfix: str = "") -> None:
        self.count = min(self.total, self.count + int(n))
        self._render(postfix=postfix)

    def close(self) -> None:
        self._render()
        print(flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample SMILES from a HF load_from_disk dataset split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--select-batch-size", type=int, default=DEFAULT_SELECT_BATCH_SIZE)
    parser.add_argument("--smiles-column", type=str, default=DEFAULT_SMILES_COLUMN)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0.")
    if args.select_batch_size <= 0:
        raise ValueError("--select-batch-size must be > 0.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be > 0.")


def _init_worker(dataset_path: str, split: str, smiles_column: str) -> None:
    global _WORKER_SPLIT_DS, _WORKER_SMILES_COLUMN
    split_ds, _resolved_from, _mode = _resolve_split_dataset(dataset_path, split)
    _WORKER_SPLIT_DS = split_ds
    _WORKER_SMILES_COLUMN = smiles_column


def _process_batch(task: tuple[int, np.ndarray, str]) -> tuple[int, int, str]:
    batch_id, batch_indices, batch_file = task
    batch_ds = _WORKER_SPLIT_DS.select(batch_indices.tolist())
    smiles_list = batch_ds[_WORKER_SMILES_COLUMN]
    with open(batch_file, "w", encoding="utf-8") as f:
        for smi in smiles_list:
            f.write(f"{smi}\n")
    return batch_id, len(smiles_list), batch_file


def _resolve_split_dataset(dataset_path: str, split: str) -> tuple[Dataset, str, str]:
    """Resolve dataset path and split for both DatasetDict and single Dataset layouts."""
    path = Path(dataset_path)
    candidates = [path]
    split_subdir = path / split
    if split_subdir != path:
        candidates.append(split_subdir)

    load_errors: list[str] = []
    for cand in candidates:
        try:
            obj = load_from_disk(str(cand))
        except Exception as e:  # keep fallback robust for mixed on-disk layouts
            load_errors.append(f"{cand}: {type(e).__name__}: {e}")
            continue

        if isinstance(obj, DatasetDict):
            if split not in obj:
                raise KeyError(
                    f"Split '{split}' not found in DatasetDict loaded from {cand}. "
                    f"Available splits: {list(obj.keys())}"
                )
            return obj[split], str(cand), "dataset_dict"

        if isinstance(obj, Dataset):
            return obj, str(cand), "dataset"

    joined = "\n".join(load_errors[-2:]) if load_errors else "no details"
    raise FileNotFoundError(
        f"Failed to load dataset from '{dataset_path}' as DatasetDict or Dataset.\n"
        f"Tried: {path} and {split_subdir}\n"
        f"Recent errors:\n{joined}"
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)

    start_time = time.time()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset from disk: {dataset_path}")
    split_ds, resolved_from, resolved_mode = _resolve_split_dataset(str(dataset_path), args.split)
    print(f"[INFO] Resolved dataset mode={resolved_mode}, loaded_from={resolved_from}")
    if args.smiles_column not in split_ds.column_names:
        raise KeyError(
            f"Column '{args.smiles_column}' not found in split '{args.split}'. "
            f"Available columns: {split_ds.column_names}"
        )

    num_rows = split_ds.num_rows
    n = args.num_samples
    if n > num_rows:
        raise ValueError(
            f"--num-samples ({n}) cannot exceed split size ({num_rows}) for split '{args.split}'."
        )

    print(
        f"[INFO] split={args.split} num_rows={num_rows} "
        f"num_samples={n} seed={args.seed} smiles_column={args.smiles_column}"
    )
    print(
        f"[INFO] Sampling indices without replacement using numpy.random.default_rng({args.seed})"
    )
    sample_start = time.time()
    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(num_rows, size=n, replace=False)
    print(f"[INFO] Index sampling done in {time.time() - sample_start:.2f}s")

    total_batches = int(math.ceil(n / args.select_batch_size))
    written = 0

    print(
        f"[INFO] Writing sampled SMILES to: {output_path} "
        f"(select_batch_size={args.select_batch_size}, batches={total_batches}, "
        f"num_workers={args.num_workers})"
    )
    if args.num_workers == 1:
        with output_path.open("w", encoding="utf-8") as f:
            pbar = ProgressBar(total=total_batches, desc="Writing batches")
            for b in range(total_batches):
                start = b * args.select_batch_size
                end = min((b + 1) * args.select_batch_size, n)
                batch_indices = sampled_indices[start:end].tolist()

                batch_ds = split_ds.select(batch_indices)
                for smi in batch_ds[args.smiles_column]:
                    f.write(f"{smi}\n")

                written += len(batch_indices)
                pbar.update(1, postfix=f"written={written}/{n}")
            pbar.close()
    else:
        temp_dir = Path(
            tempfile.mkdtemp(prefix="sample_train_10m_", dir=str(output_path.parent))
        )
        try:
            tasks: list[tuple[int, np.ndarray, str]] = []
            for b in range(total_batches):
                start = b * args.select_batch_size
                end = min((b + 1) * args.select_batch_size, n)
                batch_indices = sampled_indices[start:end]
                batch_file = temp_dir / f"batch_{b:06d}.smi"
                tasks.append((b, batch_indices, str(batch_file)))

            with cf.ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=_init_worker,
                initargs=(str(dataset_path), args.split, args.smiles_column),
            ) as executor:
                pbar_process = ProgressBar(total=total_batches, desc="Processing batches")
                for _batch_id, count, _batch_file in executor.map(_process_batch, tasks):
                    written += count
                    pbar_process.update(1, postfix=f"written={written}/{n}")
                pbar_process.close()

            with output_path.open("w", encoding="utf-8") as out_f:
                pbar_merge = ProgressBar(total=total_batches, desc="Merging batches")
                for b in range(total_batches):
                    batch_file = temp_dir / f"batch_{b:06d}.smi"
                    with batch_file.open("r", encoding="utf-8") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    pbar_merge.update(1)
                pbar_merge.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    print("[INFO] Done.")
    print(f"[SUMMARY] dataset_path={dataset_path}")
    print(f"[SUMMARY] split={args.split}")
    print(f"[SUMMARY] num_rows={num_rows}")
    print(f"[SUMMARY] num_samples={n}")
    print(f"[SUMMARY] seed={args.seed}")
    print(f"[SUMMARY] output={output_path}")
    print(f"[SUMMARY] lines_written={written}")
    print(f"[SUMMARY] num_workers={args.num_workers}")
    print(f"[SUMMARY] elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
