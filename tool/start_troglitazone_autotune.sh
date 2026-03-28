#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash tool/start_troglitazone_autotune.sh [options]

Options:
  --output-dir <dir>         Output root (default: results/pmo/autotune_troglitazone_<timestamp>)
  --gpus <csv>               GPU ids (default: 0,1,2,3,4,5,6,7)
  --slots-per-gpu <int>      Concurrent jobs per GPU (default: 2)
  --time-budget-hours <num>  Walltime budget for scheduling new trials (default: 12)
  --max-trials <int>         Total trial cap (default: 96)
  --seed <int>               run_pmo_mcts seed (default: 42)
  --conda-env <name>         Conda env (default: softmol)
  --resume                   Resume an existing tuner output dir
  --dry-run                  Preview commands only
  -h, --help                 Show this help

Notes:
  - PMO budget is fixed inside tuner: max_oracle_calls=10000, freq_log=100.
  - Oracle is fixed to troglitazone_rediscovery by default.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/pmo/autotune_troglitazone_${TS}"
GPUS="0,1,2,3,4,5,6,7"
SLOTS_PER_GPU=2
TIME_BUDGET_HOURS=12
MAX_TRIALS=96
SEED=42
CONDA_ENV="softmol"
RESUME=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --slots-per-gpu)
      SLOTS_PER_GPU="$2"
      shift 2
      ;;
    --time-budget-hours)
      TIME_BUDGET_HOURS="$2"
      shift 2
      ;;
    --max-trials)
      MAX_TRIALS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$PROJECT_ROOT"

OUT_ABS="$PROJECT_ROOT/$OUTPUT_DIR"
mkdir -p "$OUT_ABS"
LOG_PATH="$OUT_ABS/tuner_stdout.log"
PID_PATH="$OUT_ABS/tuner.pid"

CMD=(
  conda run --no-capture-output -n "$CONDA_ENV" python -u tool/autotune_pmo_troglitazone.py
  --output_root "$OUTPUT_DIR"
  --oracle_name troglitazone_rediscovery
  --gpus "$GPUS"
  --slots_per_gpu "$SLOTS_PER_GPU"
  --seed "$SEED"
  --time_budget_hours "$TIME_BUDGET_HOURS"
  --max_trials "$MAX_TRIALS"
)

if [[ "$RESUME" -eq 1 ]]; then
  CMD+=(--resume)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry_run)
  echo "[DRY-RUN] ${CMD[*]}"
  "${CMD[@]}"
  exit 0
fi

# Use setsid to fully detach from the invoking shell/session.
setsid "${CMD[@]}" >"$LOG_PATH" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_PATH"

echo "[OK] Tuner started."
echo "[INFO] PID: $PID"
echo "[INFO] Output: $OUT_ABS"
echo "[INFO] Stdout log: $LOG_PATH"
echo "[INFO] Leaderboard: $OUT_ABS/leaderboard.csv"
echo "[INFO] Trials index: $OUT_ABS/all_trials.jsonl"
echo "[INFO] State: $OUT_ABS/state.json"
echo "[INFO] Tail log: tail -f '$LOG_PATH'"
echo "[INFO] Stop tuner: kill $PID"
