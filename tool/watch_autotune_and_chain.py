#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPACE: dict[str, list[Any]] = {
    "block_size": [2, 4],
    "nucleus": [0.90, 0.95, 1.00],
    "temperature": [0.90, 1.00, 1.10, 1.20, 1.30],
    "init_children": [12, 16, 20, 24, 32],
    "n_total_children": [4, 6, 8, 10, 12],
    "c_param": [1.0, 1.6, 2.1, 2.6, 3.2, 4.0],
    "width_increase_factor": [1, 2, 3],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch one autotune run and auto-launch next focused round.")
    parser.add_argument("--current_output_dir", type=str, required=True, help="Current run directory")
    parser.add_argument("--poll_seconds", type=int, default=120)
    parser.add_argument("--summary_every_seconds", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=12, help="Top-k successful trials for analysis")
    parser.add_argument("--chain_rounds", type=int, default=1, help="How many next rounds to auto-launch")
    parser.add_argument("--next_time_budget_hours", type=float, default=10.0)
    parser.add_argument("--next_max_trials", type=int, default=72)
    parser.add_argument("--next_initial_random_trials", type=int, default=12)
    parser.add_argument("--next_exploit_probability", type=float, default=0.8)
    parser.add_argument("--next_top_pool", type=int, default=6)
    parser.add_argument("--conda_env", type=str, default="softmol")
    parser.add_argument("--oracle_name", type=str, default="troglitazone_rediscovery")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=str, default=None, help="Override GPUs; default from current state")
    parser.add_argument("--slots_per_gpu", type=int, default=None, help="Override slots; default from current state")
    parser.add_argument("--scheduler_seed", type=int, default=20260328)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _resolve_dir(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _objective(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row.get("auc_top10", -math.inf)),
        float(row.get("top10", -math.inf)),
        float(row.get("top1", -math.inf)),
    )


def _successful(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("status") == "success"]


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid(run_dir: Path) -> int | None:
    pid_path = run_dir / "tuner.pid"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _infer_gpus_and_slots(args: argparse.Namespace, state: dict[str, Any]) -> tuple[str, int]:
    gpus = args.gpus if args.gpus is not None else str(state.get("gpus", "0,1,2,3,4,5,6,7"))
    slots = args.slots_per_gpu if args.slots_per_gpu is not None else int(state.get("slots_per_gpu", 2))
    return gpus, int(slots)


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[Any, int]:
    out: dict[Any, int] = {}
    for r in rows:
        v = r[key]
        out[v] = out.get(v, 0) + 1
    return out


def _derive_values(base_values: list[Any], top_rows: list[dict[str, Any]], key: str) -> list[Any]:
    if not top_rows:
        return list(base_values)

    counts = _count_values(top_rows, key)
    best_val = top_rows[0][key]
    idx_map = {v: i for i, v in enumerate(base_values)}
    best_idx = idx_map[best_val]
    chosen_idx = {best_idx}

    threshold = max(2, math.ceil(len(top_rows) * 0.25))
    for v, c in counts.items():
        if c >= threshold and v in idx_map:
            chosen_idx.add(idx_map[v])

    # Add neighbors around chosen values.
    for idx in list(chosen_idx):
        if idx - 1 >= 0:
            chosen_idx.add(idx - 1)
        if idx + 1 < len(base_values):
            chosen_idx.add(idx + 1)

    # Ensure at least 2 choices when possible.
    if len(chosen_idx) < 2 and len(base_values) > 1:
        if best_idx - 1 >= 0:
            chosen_idx.add(best_idx - 1)
        elif best_idx + 1 < len(base_values):
            chosen_idx.add(best_idx + 1)

    # Keep a focused subset (max 4) near best.
    ranked_idx = sorted(chosen_idx, key=lambda i: (abs(i - best_idx), -counts.get(base_values[i], 0), i))
    keep_idx = sorted(ranked_idx[: min(4, len(ranked_idx))])
    return [base_values[i] for i in keep_idx]


def _to_csv(values: list[Any], is_float: bool) -> str:
    if is_float:
        return ",".join(f"{float(v):.2f}" for v in values)
    return ",".join(str(int(v)) for v in values)


def _analyze_and_build_next_args(
    rows: list[dict[str, Any]],
    top_k: int,
    base_space: dict[str, list[Any]],
) -> dict[str, str]:
    good = sorted(_successful(rows), key=_objective, reverse=True)
    top_rows = good[:top_k]
    if not top_rows:
        raise RuntimeError("No successful trial found in current round; cannot derive next round.")

    next_space: dict[str, list[Any]] = {
        "block_size": _derive_values(base_space["block_size"], top_rows, "block_size"),
        "nucleus": _derive_values(base_space["nucleus"], top_rows, "nucleus"),
        "temperature": _derive_values(base_space["temperature"], top_rows, "temperature"),
        "init_children": _derive_values(base_space["init_children"], top_rows, "init_children"),
        "n_total_children": _derive_values(base_space["n_total_children"], top_rows, "n_total_children"),
        "c_param": _derive_values(base_space["c_param"], top_rows, "c_param"),
        "width_increase_factor": _derive_values(
            base_space["width_increase_factor"], top_rows, "width_increase_factor"
        ),
    }

    return {
        "--block_size_values": _to_csv(next_space["block_size"], is_float=False),
        "--nucleus_values": _to_csv(next_space["nucleus"], is_float=True),
        "--temperature_values": _to_csv(next_space["temperature"], is_float=True),
        "--init_children_values": _to_csv(next_space["init_children"], is_float=False),
        "--n_total_children_values": _to_csv(next_space["n_total_children"], is_float=False),
        "--c_param_values": _to_csv(next_space["c_param"], is_float=True),
        "--width_increase_factor_values": _to_csv(next_space["width_increase_factor"], is_float=False),
    }


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _launch_next_round(
    next_dir: Path,
    args: argparse.Namespace,
    gpus: str,
    slots_per_gpu: int,
    narrowed_space_args: dict[str, str],
    round_idx: int,
) -> tuple[int, list[str]]:
    next_dir.mkdir(parents=True, exist_ok=True)
    log_path = next_dir / "tuner_stdout.log"
    pid_path = next_dir / "tuner.pid"

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "-u",
        "tool/autotune_pmo_troglitazone.py",
        "--output_root",
        _relative_to_project(next_dir),
        "--oracle_name",
        args.oracle_name,
        "--gpus",
        gpus,
        "--slots_per_gpu",
        str(slots_per_gpu),
        "--seed",
        str(args.seed),
        "--time_budget_hours",
        str(args.next_time_budget_hours),
        "--max_trials",
        str(args.next_max_trials),
        "--initial_random_trials",
        str(args.next_initial_random_trials),
        "--exploit_probability",
        str(args.next_exploit_probability),
        "--top_pool",
        str(args.next_top_pool),
        "--scheduler_seed",
        str(args.scheduler_seed + round_idx),
    ]
    for k, v in narrowed_space_args.items():
        cmd.extend([k, v])

    if args.dry_run:
        print("[DRY-RUN] next round cmd:")
        print(" ".join(cmd))
        return -1, cmd

    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_f.close()
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    return proc.pid, cmd


def _wait_until_finished(run_dir: Path, poll_seconds: int, summary_every_seconds: int) -> list[dict[str, Any]]:
    state_path = run_dir / "state.json"
    trials_path = run_dir / "all_trials.jsonl"
    pid = _read_pid(run_dir)

    if pid is None:
        raise FileNotFoundError(f"Cannot find tuner pid file: {run_dir / 'tuner.pid'}")

    last_completed = -1
    last_summary_t = 0.0

    while True:
        rows = _load_jsonl(trials_path)
        good = _successful(rows)
        completed = len(rows)
        now = time.time()

        if completed != last_completed:
            last_completed = completed
            if good:
                best = max(good, key=_objective)
                print(
                    f"[WATCH] completed={completed} success={len(good)} "
                    f"best_trial={best['trial_id']} auc_top10={float(best['auc_top10']):.6f} "
                    f"top10={float(best['top10']):.6f}"
                )
            else:
                print(f"[WATCH] completed={completed} success=0 (waiting first successful result)")

        if now - last_summary_t >= summary_every_seconds:
            if state_path.exists():
                state = _load_json(state_path)
                print(
                    f"[WATCH] state launched={state.get('launched_trials')} "
                    f"running={state.get('running_trials')} completed={state.get('completed_trials')} "
                    f"elapsed_h={float(state.get('elapsed_hours', 0.0)):.2f}"
                )
            last_summary_t = now

        if not _is_pid_alive(pid):
            print(f"[WATCH] tuner pid={pid} finished.")
            return rows

        time.sleep(poll_seconds)


def main() -> None:
    args = _parse_args()
    current_dir = _resolve_dir(args.current_output_dir)
    if not current_dir.exists():
        raise FileNotFoundError(f"Current output directory not found: {current_dir}")

    round_dir = current_dir
    for round_idx in range(1, int(args.chain_rounds) + 1):
        print(f"[WATCH] monitoring round{round_idx} dir={round_dir}")
        rows = _wait_until_finished(
            run_dir=round_dir,
            poll_seconds=int(args.poll_seconds),
            summary_every_seconds=int(args.summary_every_seconds),
        )
        good = sorted(_successful(rows), key=_objective, reverse=True)
        if not good:
            raise RuntimeError(f"No successful trial in {round_dir}; stop chaining.")

        best = good[0]
        print(
            f"[WATCH] round{round_idx} done. best trial={best['trial_id']} "
            f"auc_top10={float(best['auc_top10']):.6f} top10={float(best['top10']):.6f} "
            f"top1={float(best['top1']):.6f}"
        )

        state_path = round_dir / "state.json"
        state = _load_json(state_path) if state_path.exists() else {}
        gpus, slots_per_gpu = _infer_gpus_and_slots(args, state)
        narrowed = _analyze_and_build_next_args(
            rows=rows,
            top_k=int(args.top_k),
            base_space=DEFAULT_SPACE,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        next_dir = round_dir.parent / f"{round_dir.name}_focus_r{round_idx + 1}_{ts}"
        pid, cmd = _launch_next_round(
            next_dir=next_dir,
            args=args,
            gpus=gpus,
            slots_per_gpu=slots_per_gpu,
            narrowed_space_args=narrowed,
            round_idx=round_idx,
        )

        summary = {
            "source_round_dir": str(round_dir),
            "source_success_trials": len(good),
            "source_best_trial": int(best["trial_id"]),
            "source_best_auc_top10": float(best["auc_top10"]),
            "source_best_top10": float(best["top10"]),
            "source_best_top1": float(best["top1"]),
            "derived_space": narrowed,
            "next_round_dir": str(next_dir),
            "next_round_pid": int(pid),
            "next_round_cmd": cmd,
            "created_at": int(time.time()),
        }
        (round_dir / "next_round_plan.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if args.dry_run:
            print("[WATCH] dry-run finished, no actual next round launched.")
            return

        print(f"[WATCH] next round launched: pid={pid} dir={next_dir}")
        round_dir = next_dir

    print("[WATCH] chain completed.")


if __name__ == "__main__":
    main()
