#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixed by user requirement.
FIXED_MAX_ORACLE_CALLS = 10000
FIXED_FREQ_LOG = 100

DEFAULT_SEARCH_SPACE = {
    "block_size": [2, 4],
    "nucleus": [0.90, 0.95, 1.00],
    "temperature": [0.90, 1.00, 1.10, 1.20, 1.30],
    "init_children": [12, 16, 20, 24, 32],
    "n_total_children": [4, 6, 8, 10, 12],
    "c_param": [1.0, 1.6, 2.1, 2.6, 3.2, 4.0],
    "width_increase_factor": [1, 2, 3],
}
ACTIVE_SEARCH_SPACE = {k: list(v) for k, v in DEFAULT_SEARCH_SPACE.items()}


@dataclass
class RunningJob:
    trial_id: int
    slot_id: int
    gpu_id: int
    seed: int
    config: dict[str, Any]
    run_dir: Path
    log_path: Path
    cmd: list[str]
    proc: subprocess.Popen[Any]
    log_handle: Any
    start_time: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Asynchronous hyperparameter tuning for PMO troglitazone_rediscovery."
    )
    parser.add_argument("--output_root", type=str, required=True, help="Root directory for tuner artifacts.")
    parser.add_argument("--oracle_name", type=str, default="troglitazone_rediscovery")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots_per_gpu", type=int, default=2)
    parser.add_argument("--conda_env", type=str, default="softmol")
    parser.add_argument("--seed", type=int, default=42, help="Seed used inside run_pmo_mcts.")
    parser.add_argument("--scheduler_seed", type=int, default=20260328, help="Seed for tuner sampling.")
    parser.add_argument("--time_budget_hours", type=float, default=12.0)
    parser.add_argument("--max_trials", type=int, default=96)
    parser.add_argument("--initial_random_trials", type=int, default=24)
    parser.add_argument("--exploit_probability", type=float, default=0.65)
    parser.add_argument("--top_pool", type=int, default=8)
    parser.add_argument("--block_size_values", type=str, default=None, help="CSV overrides, e.g. 2,4")
    parser.add_argument("--nucleus_values", type=str, default=None, help="CSV overrides, e.g. 0.9,0.95,1.0")
    parser.add_argument("--temperature_values", type=str, default=None, help="CSV overrides")
    parser.add_argument("--init_children_values", type=str, default=None, help="CSV overrides")
    parser.add_argument("--n_total_children_values", type=str, default=None, help="CSV overrides")
    parser.add_argument("--c_param_values", type=str, default=None, help="CSV overrides")
    parser.add_argument("--width_increase_factor_values", type=str, default=None, help="CSV overrides")
    parser.add_argument("--poll_seconds", type=float, default=10.0)
    parser.add_argument("--status_every_seconds", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_trials", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default="weights/89M-epoch6-best.ckpt")
    parser.add_argument("--vocab", type=str, default="vocab_V2.txt")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_batch_size", type=int, default=1)
    parser.add_argument("--model", type=str, default="small-89M")
    parser.add_argument("--search_time", type=int, default=100000)
    parser.add_argument("--save_topk", type=int, default=100)
    return parser.parse_args()


def _parse_csv_values(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    vals = [int(x) for x in _parse_csv_values(raw)]
    if not vals:
        raise ValueError(f"Invalid integer CSV values: {raw}")
    return sorted(set(vals))


def _parse_float_csv(raw: str) -> list[float]:
    vals = [float(x) for x in _parse_csv_values(raw)]
    if not vals:
        raise ValueError(f"Invalid float CSV values: {raw}")
    return sorted(set(vals))


def _build_search_space(args: argparse.Namespace) -> dict[str, list[Any]]:
    space: dict[str, list[Any]] = {k: list(v) for k, v in DEFAULT_SEARCH_SPACE.items()}

    if args.block_size_values is not None:
        vals = _parse_int_csv(args.block_size_values)
        if any(v > 4 for v in vals):
            raise ValueError("block_size must be <= 4.")
        space["block_size"] = vals
    if args.nucleus_values is not None:
        space["nucleus"] = _parse_float_csv(args.nucleus_values)
    if args.temperature_values is not None:
        space["temperature"] = _parse_float_csv(args.temperature_values)
    if args.init_children_values is not None:
        space["init_children"] = _parse_int_csv(args.init_children_values)
    if args.n_total_children_values is not None:
        space["n_total_children"] = _parse_int_csv(args.n_total_children_values)
    if args.c_param_values is not None:
        space["c_param"] = _parse_float_csv(args.c_param_values)
    if args.width_increase_factor_values is not None:
        space["width_increase_factor"] = _parse_int_csv(args.width_increase_factor_values)

    for k, v in space.items():
        if not v:
            raise ValueError(f"Search space for {k} is empty.")
    return space


def _resolve_output_root(output_root: str) -> Path:
    path = Path(output_root)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _parse_gpus(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        values.append(int(t))
    if not values:
        raise ValueError(f"No valid GPU parsed from: {raw}")
    return values


def _normalize_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out["block_size"] = int(out["block_size"])
    out["nucleus"] = float(out["nucleus"])
    out["temperature"] = float(out["temperature"])
    out["init_children"] = int(out["init_children"])
    out["n_total_children"] = int(out["n_total_children"])
    out["c_param"] = float(out["c_param"])
    out["width_increase_factor"] = int(out["width_increase_factor"])

    # Keep root branching at least as wide as non-root branching.
    if out["init_children"] < out["n_total_children"]:
        valid = [v for v in ACTIVE_SEARCH_SPACE["init_children"] if v >= out["n_total_children"]]
        out["init_children"] = valid[0] if valid else max(ACTIVE_SEARCH_SPACE["init_children"])
    return out


def _cfg_key(cfg: dict[str, Any]) -> str:
    c = _normalize_cfg(cfg)
    return (
        f"bs={c['block_size']};p={c['nucleus']:.2f};t={c['temperature']:.2f};"
        f"init={c['init_children']};child={c['n_total_children']};"
        f"c={c['c_param']:.2f};w={c['width_increase_factor']}"
    )


def _sample_random_cfg(rng: random.Random) -> dict[str, Any]:
    cfg = {
        "block_size": rng.choice(ACTIVE_SEARCH_SPACE["block_size"]),
        "nucleus": rng.choice(ACTIVE_SEARCH_SPACE["nucleus"]),
        "temperature": rng.choice(ACTIVE_SEARCH_SPACE["temperature"]),
        "init_children": rng.choice(ACTIVE_SEARCH_SPACE["init_children"]),
        "n_total_children": rng.choice(ACTIVE_SEARCH_SPACE["n_total_children"]),
        "c_param": rng.choice(ACTIVE_SEARCH_SPACE["c_param"]),
        "width_increase_factor": rng.choice(ACTIVE_SEARCH_SPACE["width_increase_factor"]),
    }
    return _normalize_cfg(cfg)


def _neighbor_choice(values: list[Any], current: Any, rng: random.Random) -> Any:
    idx = values.index(current)
    candidates = [idx]
    if idx - 1 >= 0:
        candidates.append(idx - 1)
    if idx + 1 < len(values):
        candidates.append(idx + 1)
    return values[rng.choice(candidates)]


def _mutate_cfg(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    cfg = _normalize_cfg(base)
    out = dict(cfg)
    changed = False

    for key, values in ACTIVE_SEARCH_SPACE.items():
        if rng.random() < 0.45:
            new_val = _neighbor_choice(values, cfg[key], rng)
            out[key] = new_val
            changed = changed or (new_val != cfg[key])

    if not changed:
        pick_key = rng.choice(list(ACTIVE_SEARCH_SPACE.keys()))
        out[pick_key] = _neighbor_choice(ACTIVE_SEARCH_SPACE[pick_key], cfg[pick_key], rng)

    return _normalize_cfg(out)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_metrics(run_dir: Path, oracle_name: str, seed: int) -> dict[str, Any] | None:
    metrics_path = run_dir / f"{oracle_name}_seed{seed}_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _objective(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics.get("auc_top10", -math.inf)),
        float(metrics.get("top10", -math.inf)),
        float(metrics.get("top1", -math.inf)),
    )


def _successful_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") == "success":
            out.append(row)
    return out


def _write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    good = _successful_records(rows)
    good_sorted = sorted(
        good,
        key=lambda x: (
            float(x.get("auc_top10", -math.inf)),
            float(x.get("top10", -math.inf)),
            float(x.get("top1", -math.inf)),
        ),
        reverse=True,
    )
    fields = [
        "rank",
        "trial_id",
        "auc_top10",
        "top10",
        "top1",
        "elapsed_time_sec",
        "gpu_id",
        "slot_id",
        "seed",
        "block_size",
        "nucleus",
        "temperature",
        "init_children",
        "n_total_children",
        "c_param",
        "width_increase_factor",
        "run_dir",
        "log_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(good_sorted, start=1):
            item = {k: row.get(k, "") for k in fields}
            item["rank"] = rank
            writer.writerow(item)


def _build_cmd(args: argparse.Namespace, run_dir: Path, cfg: dict[str, Any], gpu_id: int, seed: int) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "gated_mcts/run_pmo_mcts.py",
        "--seed",
        str(seed),
        "--device",
        str(gpu_id),
        "--max_oracle_calls",
        str(FIXED_MAX_ORACLE_CALLS),
        "--freq_log",
        str(FIXED_FREQ_LOG),
        "--output_dir",
        str(run_dir),
        "--oracle_name",
        str(args.oracle_name),
        "--ckpt",
        str(args.ckpt),
        "--vocab",
        str(args.vocab),
        "--length",
        str(args.length),
        "--steps",
        str(args.steps),
        "--gen_batch_size",
        str(args.gen_batch_size),
        "--model",
        str(args.model),
        "--search_time",
        str(args.search_time),
        "--save_topk",
        str(args.save_topk),
        "--block_size",
        str(cfg["block_size"]),
        "--nucleus",
        str(cfg["nucleus"]),
        "--temperature",
        str(cfg["temperature"]),
        "--init_children",
        str(cfg["init_children"]),
        "--n_total_children",
        str(cfg["n_total_children"]),
        "--c_param",
        str(cfg["c_param"]),
        "--width_increase_factor",
        str(cfg["width_increase_factor"]),
    ]


def _select_next_cfg(
    rng: random.Random,
    tried: set[str],
    success_rows: list[dict[str, Any]],
    initial_random_trials: int,
    exploit_probability: float,
    top_pool: int,
) -> dict[str, Any] | None:
    ranked = sorted(
        success_rows,
        key=lambda row: (
            float(row.get("auc_top10", -math.inf)),
            float(row.get("top10", -math.inf)),
            float(row.get("top1", -math.inf)),
        ),
        reverse=True,
    )

    for _ in range(400):
        if len(success_rows) < initial_random_trials or not ranked or rng.random() > exploit_probability:
            cfg = _sample_random_cfg(rng)
        else:
            parent_row = rng.choice(ranked[: max(1, min(top_pool, len(ranked)))])
            parent_cfg = {
                "block_size": int(parent_row["block_size"]),
                "nucleus": float(parent_row["nucleus"]),
                "temperature": float(parent_row["temperature"]),
                "init_children": int(parent_row["init_children"]),
                "n_total_children": int(parent_row["n_total_children"]),
                "c_param": float(parent_row["c_param"]),
                "width_increase_factor": int(parent_row["width_increase_factor"]),
            }
            cfg = _mutate_cfg(parent_cfg, rng)
        key = _cfg_key(cfg)
        if key not in tried:
            return cfg
    return None


def _state_row(
    args: argparse.Namespace,
    next_trial_id: int,
    launched: int,
    running: int,
    completed: int,
    start_time: float,
) -> dict[str, Any]:
    return {
        "oracle_name": args.oracle_name,
        "fixed_max_oracle_calls": FIXED_MAX_ORACLE_CALLS,
        "fixed_freq_log": FIXED_FREQ_LOG,
        "gpus": args.gpus,
        "slots_per_gpu": int(args.slots_per_gpu),
        "time_budget_hours": float(args.time_budget_hours),
        "max_trials": int(args.max_trials),
        "next_trial_id": int(next_trial_id),
        "launched_trials": int(launched),
        "running_trials": int(running),
        "completed_trials": int(completed),
        "elapsed_hours": float((time.time() - start_time) / 3600.0),
        "updated_at": int(time.time()),
    }


def _print_startup(args: argparse.Namespace, slots: list[tuple[int, int]], out_root: Path) -> None:
    print("[TUNER] start")
    print(f"[TUNER] project_root={PROJECT_ROOT}")
    print(f"[TUNER] output_root={out_root}")
    print(f"[TUNER] oracle={args.oracle_name}")
    print(f"[TUNER] fixed max_oracle_calls={FIXED_MAX_ORACLE_CALLS}, freq_log={FIXED_FREQ_LOG}")
    print(f"[TUNER] gpus={args.gpus}, slots_per_gpu={args.slots_per_gpu}, total_slots={len(slots)}")
    print(
        f"[TUNER] time_budget_hours={args.time_budget_hours}, max_trials={args.max_trials}, "
        f"initial_random_trials={args.initial_random_trials}"
    )
    print(
        f"[TUNER] search space sizes: "
        f"block_size={len(ACTIVE_SEARCH_SPACE['block_size'])} "
        f"nucleus={len(ACTIVE_SEARCH_SPACE['nucleus'])} "
        f"temperature={len(ACTIVE_SEARCH_SPACE['temperature'])} "
        f"init_children={len(ACTIVE_SEARCH_SPACE['init_children'])} "
        f"n_total_children={len(ACTIVE_SEARCH_SPACE['n_total_children'])} "
        f"c_param={len(ACTIVE_SEARCH_SPACE['c_param'])} "
        f"width_increase_factor={len(ACTIVE_SEARCH_SPACE['width_increase_factor'])}"
    )
    print("[TUNER] objective=auc_top10 (tie-break: top10, top1)")
    print()


def main() -> None:
    global ACTIVE_SEARCH_SPACE
    args = _parse_args()
    ACTIVE_SEARCH_SPACE = _build_search_space(args)
    out_root = _resolve_output_root(args.output_root)
    runs_dir = out_root / "runs"
    logs_dir = out_root / "logs"
    state_path = out_root / "state.json"
    all_trials_path = out_root / "all_trials.jsonl"
    leaderboard_path = out_root / "leaderboard.csv"

    gpus = _parse_gpus(args.gpus)
    slots: list[tuple[int, int]] = []
    for gpu in gpus:
        for slot_idx in range(int(args.slots_per_gpu)):
            slots.append((gpu, slot_idx))
    if not slots:
        raise ValueError("No execution slot available.")

    if out_root.exists() and not args.resume and not args.dry_run:
        allowed_bootstrap = {"tuner_stdout.log", "tuner.pid"}
        has_conflict = any(p.name not in allowed_bootstrap for p in out_root.iterdir())
        if has_conflict:
            raise RuntimeError(f"Output root already exists and has tuner artifacts: {out_root}. Use --resume.")

    out_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(all_trials_path) if args.resume else []
    tried: set[str] = set()
    next_trial_id = 1
    for row in rows:
        cfg = {
            "block_size": int(row["block_size"]),
            "nucleus": float(row["nucleus"]),
            "temperature": float(row["temperature"]),
            "init_children": int(row["init_children"]),
            "n_total_children": int(row["n_total_children"]),
            "c_param": float(row["c_param"]),
            "width_increase_factor": int(row["width_increase_factor"]),
        }
        tried.add(_cfg_key(cfg))
        next_trial_id = max(next_trial_id, int(row.get("trial_id", 0)) + 1)

    rng = random.Random(args.scheduler_seed)
    _print_startup(args, slots, out_root)

    if args.dry_run:
        print("[TUNER] dry-run preview")
        preview_rows = _successful_records(rows)
        for i in range(int(args.dry_run_trials)):
            cfg = _select_next_cfg(
                rng=rng,
                tried=tried,
                success_rows=preview_rows,
                initial_random_trials=int(args.initial_random_trials),
                exploit_probability=float(args.exploit_probability),
                top_pool=int(args.top_pool),
            )
            if cfg is None:
                print("[TUNER] no more unique configuration found")
                break
            trial_id = next_trial_id + i
            gpu_id, slot_id = slots[i % len(slots)]
            run_dir = runs_dir / f"trial_{trial_id:04d}"
            cmd = _build_cmd(args, run_dir, cfg, gpu_id, int(args.seed))
            tried.add(_cfg_key(cfg))
            print(f"[DRY-RUN] trial={trial_id} gpu={gpu_id} slot={slot_id} cfg={cfg}")
            print(f"          cmd={' '.join(shlex.quote(x) for x in cmd)}")
        return

    running: dict[int, RunningJob] = {}
    start_wall = time.time()
    deadline = start_wall + float(args.time_budget_hours) * 3600.0
    launched = len(rows)
    last_status_time = 0.0
    shutting_down = False

    try:
        while True:
            now = time.time()
            can_launch = now < deadline and launched < int(args.max_trials) and not shutting_down

            # Fill free slots.
            if can_launch:
                for slot_id, (gpu_id, slot_idx) in enumerate(slots):
                    if slot_id in running:
                        continue
                    if launched >= int(args.max_trials):
                        break
                    cfg = _select_next_cfg(
                        rng=rng,
                        tried=tried,
                        success_rows=_successful_records(rows),
                        initial_random_trials=int(args.initial_random_trials),
                        exploit_probability=float(args.exploit_probability),
                        top_pool=int(args.top_pool),
                    )
                    if cfg is None:
                        can_launch = False
                        break

                    trial_id = next_trial_id
                    next_trial_id += 1
                    launched += 1
                    tried.add(_cfg_key(cfg))

                    run_dir = runs_dir / f"trial_{trial_id:04d}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    log_path = logs_dir / f"trial_{trial_id:04d}.log"
                    cmd = _build_cmd(args, run_dir, cfg, gpu_id, int(args.seed))

                    log_f = open(log_path, "w", encoding="utf-8")
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                    )
                    running[slot_id] = RunningJob(
                        trial_id=trial_id,
                        slot_id=slot_idx,
                        gpu_id=gpu_id,
                        seed=int(args.seed),
                        config=cfg,
                        run_dir=run_dir,
                        log_path=log_path,
                        cmd=cmd,
                        proc=proc,
                        log_handle=log_f,
                        start_time=time.time(),
                    )
                    print(
                        f"[LAUNCH] trial={trial_id} gpu={gpu_id} slot={slot_idx} pid={proc.pid} "
                        f"cfg={cfg}"
                    )

            # Check completed jobs.
            finished_slot_ids: list[int] = []
            for slot_id, job in running.items():
                ret = job.proc.poll()
                if ret is None:
                    continue
                finished_slot_ids.append(slot_id)
                job.log_handle.close()
                end_time = time.time()
                metrics = _load_metrics(job.run_dir, args.oracle_name, job.seed)
                status = "success" if (ret == 0 and metrics is not None) else "failed"
                row: dict[str, Any] = {
                    "trial_id": int(job.trial_id),
                    "status": status,
                    "return_code": int(ret),
                    "start_time": float(job.start_time),
                    "end_time": float(end_time),
                    "elapsed_time_sec": float(end_time - job.start_time),
                    "gpu_id": int(job.gpu_id),
                    "slot_id": int(job.slot_id),
                    "seed": int(job.seed),
                    "block_size": int(job.config["block_size"]),
                    "nucleus": float(job.config["nucleus"]),
                    "temperature": float(job.config["temperature"]),
                    "init_children": int(job.config["init_children"]),
                    "n_total_children": int(job.config["n_total_children"]),
                    "c_param": float(job.config["c_param"]),
                    "width_increase_factor": int(job.config["width_increase_factor"]),
                    "run_dir": str(job.run_dir),
                    "log_path": str(job.log_path),
                    "metrics_path": str(job.run_dir / f"{args.oracle_name}_seed{job.seed}_metrics.json"),
                    "cmd": " ".join(shlex.quote(x) for x in job.cmd),
                }
                if metrics is not None:
                    row["n_calls"] = int(metrics.get("n_calls", 0))
                    row["auc_top10"] = float(metrics.get("auc_top10", float("nan")))
                    row["top10"] = float(metrics.get("top10", float("nan")))
                    row["top1"] = float(metrics.get("top1", float("nan")))
                    row["best_smi"] = metrics.get("best_smi", "")
                rows.append(row)
                _append_jsonl(all_trials_path, row)
                _write_leaderboard(leaderboard_path, rows)

                if status == "success":
                    print(
                        f"[DONE] trial={job.trial_id} auc_top10={row['auc_top10']:.6f} "
                        f"top10={row['top10']:.6f} top1={row['top1']:.6f} "
                        f"elapsed={row['elapsed_time_sec'] / 60.0:.1f}m"
                    )
                else:
                    print(
                        f"[FAIL] trial={job.trial_id} ret={ret} "
                        f"log={job.log_path}"
                    )

            for slot_id in finished_slot_ids:
                running.pop(slot_id, None)

            # Persist state snapshot.
            _write_json(
                state_path,
                _state_row(
                    args=args,
                    next_trial_id=next_trial_id,
                    launched=launched,
                    running=len(running),
                    completed=len(rows),
                    start_time=start_wall,
                ),
            )

            # Periodic status.
            now = time.time()
            if now - last_status_time >= float(args.status_every_seconds):
                good = _successful_records(rows)
                best_msg = "none"
                if good:
                    best = max(good, key=lambda r: _objective(r))
                    best_msg = (
                        f"trial={best['trial_id']} auc_top10={float(best['auc_top10']):.6f} "
                        f"top10={float(best['top10']):.6f}"
                    )
                print(
                    f"[STATUS] launched={launched}/{args.max_trials} "
                    f"running={len(running)} completed={len(rows)} "
                    f"elapsed_h={(now - start_wall) / 3600.0:.2f} best={best_msg}"
                )
                last_status_time = now

            # Exit conditions.
            time_up = time.time() >= deadline
            done_launch = launched >= int(args.max_trials) or time_up
            if done_launch and not running:
                break

            time.sleep(float(args.poll_seconds))

    except KeyboardInterrupt:
        shutting_down = True
        print("[TUNER] KeyboardInterrupt received, terminating running jobs...")
        for job in running.values():
            try:
                job.proc.terminate()
            except Exception:
                pass
        time.sleep(3)
        for job in running.values():
            if job.proc.poll() is None:
                try:
                    job.proc.kill()
                except Exception:
                    pass
        for job in running.values():
            try:
                job.log_handle.close()
            except Exception:
                pass
        raise

    good = _successful_records(rows)
    print()
    print("[TUNER] finished")
    print(f"[TUNER] output_root={out_root}")
    print(f"[TUNER] total_trials={len(rows)} success={len(good)} failed={len(rows) - len(good)}")
    if good:
        best = max(good, key=lambda r: _objective(r))
        print(
            f"[TUNER] best trial={best['trial_id']} auc_top10={float(best['auc_top10']):.6f} "
            f"top10={float(best['top10']):.6f} top1={float(best['top1']):.6f}"
        )
        print(f"[TUNER] best run_dir={best['run_dir']}")
    print(f"[TUNER] leaderboard={leaderboard_path}")
    print(f"[TUNER] all_trials={all_trials_path}")


if __name__ == "__main__":
    main()
