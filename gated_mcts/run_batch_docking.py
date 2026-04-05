from __future__ import annotations

import argparse
import csv
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rdkit import Chem, RDLogger

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gated_mcts.utils.docking.docking_utils import DockingVina, SUPPORTED_TARGETS

RDLogger.DisableLog("rdApp.*")

DEFAULT_TARGETS = ["parp1", "fa7", "5ht1b", "braf", "jak2"]
FAIL_SCORE = 1.0


def _chunks(seq: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _read_input_smiles(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines]


def _build_unique_mapping(smiles_list: List[str]) -> Tuple[List[str], List[int]]:
    unique_smiles: List[str] = []
    orig_to_unique: List[int] = []
    seen: Dict[str, int] = {}
    for smi in smiles_list:
        if smi in seen:
            orig_to_unique.append(seen[smi])
            continue
        idx = len(unique_smiles)
        seen[smi] = idx
        unique_smiles.append(smi)
        orig_to_unique.append(idx)
    return unique_smiles, orig_to_unique


def _validate_unique_smiles(unique_smiles: List[str]) -> List[bool]:
    valid_mask: List[bool] = []
    for smi in unique_smiles:
        if not smi:
            valid_mask.append(False)
            continue
        mol = Chem.MolFromSmiles(smi)
        valid_mask.append(mol is not None)
    return valid_mask


def _target_checkpoint_path(output_csv: Path, target: str) -> Path:
    base = output_csv.stem
    if base.endswith("_5targets"):
        base = base[: -len("_5targets")]
    return output_csv.parent / f"{base}_{target}.csv"


def _load_checkpoint(path: Path) -> Dict[int, float]:
    if not path.exists():
        return {}
    out: Dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["unique_idx"])
                affinity = float(row["affinity"])
            except Exception:
                continue
            out[idx] = affinity
    return out


def _append_checkpoint_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = ["unique_idx", "smiles", "affinity", "source"]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _run_target_docking(
    target: str,
    unique_smiles: List[str],
    valid_mask: List[bool],
    checkpoint_path: str,
    batch_size: int,
    cpu_per_dock: int,
    resume: bool,
) -> Tuple[str, Dict[int, float], dict]:
    ckpt = Path(checkpoint_path)
    if not resume and ckpt.exists():
        ckpt.unlink()

    scores = _load_checkpoint(ckpt) if resume else {}
    start_known = len(scores)
    n_unique = len(unique_smiles)

    buffered_rows: List[dict] = []
    invalid_count = 0
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            continue
        if i in scores:
            continue
        scores[i] = FAIL_SCORE
        invalid_count += 1
        buffered_rows.append(
            {
                "unique_idx": i,
                "smiles": unique_smiles[i],
                "affinity": FAIL_SCORE,
                "source": "invalid_smiles",
            }
        )
    _append_checkpoint_rows(ckpt, buffered_rows)

    pending = [i for i in range(n_unique) if (valid_mask[i] and i not in scores)]
    total_batches = (len(pending) + batch_size - 1) // batch_size if pending else 0
    docking_fails = 0

    if pending:
        predictor = DockingVina(target)
        predictor.num_cpu_dock = int(cpu_per_dock)

        for bidx, batch_idxs in enumerate(_chunks(pending, batch_size), start=1):
            batch_smiles = [unique_smiles[i] for i in batch_idxs]
            try:
                affinities = predictor.predict(batch_smiles)
            except Exception as e:
                print(
                    f"[{target}] batch {bidx}/{total_batches} failed ({type(e).__name__}), "
                    f"fallback to {FAIL_SCORE} for {len(batch_idxs)} molecules"
                )
                affinities = [FAIL_SCORE] * len(batch_idxs)

            rows: List[dict] = []
            for i, affinity in zip(batch_idxs, affinities):
                try:
                    val = float(affinity)
                except Exception:
                    val = FAIL_SCORE
                if not math.isfinite(val):
                    val = FAIL_SCORE
                if val == FAIL_SCORE:
                    docking_fails += 1
                scores[i] = val
                rows.append(
                    {
                        "unique_idx": i,
                        "smiles": unique_smiles[i],
                        "affinity": val,
                        "source": "docking",
                    }
                )

            _append_checkpoint_rows(ckpt, rows)
            print(
                f"[{target}] batch {bidx}/{total_batches} done, "
                f"known={len(scores)}/{n_unique}, pending={n_unique - len(scores)}"
            )

    # Fill any missing entries defensively to keep downstream table complete.
    missing = [i for i in range(n_unique) if i not in scores]
    if missing:
        rows = []
        for i in missing:
            scores[i] = FAIL_SCORE
            rows.append(
                {
                    "unique_idx": i,
                    "smiles": unique_smiles[i],
                    "affinity": FAIL_SCORE,
                    "source": "missing_fallback",
                }
            )
        _append_checkpoint_rows(ckpt, rows)

    stats = {
        "target": target,
        "n_unique": n_unique,
        "start_known": start_known,
        "new_invalid": invalid_count,
        "new_docking": len(pending),
        "docking_fail": docking_fails,
        "final_fail": sum(1 for x in scores.values() if float(x) == FAIL_SCORE),
        "checkpoint": str(ckpt),
    }
    return target, scores, stats


def _write_final_output(
    output_path: Path,
    input_smiles: List[str],
    orig_to_unique: List[int],
    targets: List[str],
    target_to_scores: Dict[str, Dict[int, float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "smiles"] + [f"{t}_affinity" for t in targets]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ridx, smi in enumerate(input_smiles):
            uidx = orig_to_unique[ridx]
            row = {"row_id": ridx + 1, "smiles": smi}
            for t in targets:
                row[f"{t}_affinity"] = float(target_to_scores[t][uidx])
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch qvina docking for a SMILES list across multiple targets.")
    parser.add_argument("--input", type=str, default="results/denovo.txt", help="Input SMILES txt file (one SMILES per line)")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help=f"Docking targets (supported: {', '.join(sorted(SUPPORTED_TARGETS))})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/denovo_qvina_affinity_5targets.csv",
        help="Final output CSV path",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size over unique SMILES")
    parser.add_argument("--parallel_targets", type=int, default=2, help="How many targets to process in parallel")
    parser.add_argument("--cpu_per_dock", type=int, default=5, help="Docking CPU count passed to qvina")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from per-target checkpoint CSVs (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    targets = [str(t).strip() for t in args.targets if str(t).strip()]
    if not targets:
        raise ValueError("No valid target provided.")
    unsupported = [t for t in targets if t not in SUPPORTED_TARGETS]
    if unsupported:
        raise ValueError(f"Unsupported targets: {unsupported}. Supported: {sorted(SUPPORTED_TARGETS)}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.parallel_targets <= 0:
        raise ValueError("--parallel_targets must be > 0")
    if args.cpu_per_dock <= 0:
        raise ValueError("--cpu_per_dock must be > 0")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_smiles = _read_input_smiles(input_path)
    unique_smiles, orig_to_unique = _build_unique_mapping(input_smiles)
    valid_mask = _validate_unique_smiles(unique_smiles)
    n_invalid = sum(1 for v in valid_mask if not v)
    print(
        f"Loaded input: total_rows={len(input_smiles)}, unique={len(unique_smiles)}, "
        f"duplicates={len(input_smiles) - len(unique_smiles)}, invalid_unique={n_invalid}"
    )

    checkpoint_paths = {t: _target_checkpoint_path(output_path, t) for t in targets}
    for t, ckpt in checkpoint_paths.items():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{t}] checkpoint: {ckpt}")

    target_to_scores: Dict[str, Dict[int, float]] = {}
    target_stats: Dict[str, dict] = {}

    max_workers = min(int(args.parallel_targets), len(targets))
    if max_workers == 1:
        for t in targets:
            tt, scores, stats = _run_target_docking(
                target=t,
                unique_smiles=unique_smiles,
                valid_mask=valid_mask,
                checkpoint_path=str(checkpoint_paths[t]),
                batch_size=int(args.batch_size),
                cpu_per_dock=int(args.cpu_per_dock),
                resume=bool(args.resume),
            )
            target_to_scores[tt] = scores
            target_stats[tt] = stats
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _run_target_docking,
                    t,
                    unique_smiles,
                    valid_mask,
                    str(checkpoint_paths[t]),
                    int(args.batch_size),
                    int(args.cpu_per_dock),
                    bool(args.resume),
                ): t
                for t in targets
            }
            for fut in as_completed(futures):
                tt, scores, stats = fut.result()
                target_to_scores[tt] = scores
                target_stats[tt] = stats

    _write_final_output(
        output_path=output_path,
        input_smiles=input_smiles,
        orig_to_unique=orig_to_unique,
        targets=targets,
        target_to_scores=target_to_scores,
    )

    print(f"Final output saved: {output_path}")
    print("Failure summary (affinity == 1.0):")
    for t in targets:
        st = target_stats[t]
        print(
            f"  - {t}: final_fail={st['final_fail']} "
            f"(new_invalid={st['new_invalid']}, new_docking_fail={st['docking_fail']}, "
            f"new_docking_called={st['new_docking']}, start_known={st['start_known']})"
        )


if __name__ == "__main__":
    main()
