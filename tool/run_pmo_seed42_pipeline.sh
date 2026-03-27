#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash tool/run_pmo_seed42_pipeline.sh [options]

Options:
  --output-dir <dir>         PMO output dir (default: results/pmo/softmol_mcts_seed42_20260327)
  --seed <int>               Seed (default: 42)
  --gpus <csv>               GPUs (default: 0,1,2,3,4,5,6,7)
  --conda-env <name>         Conda env (default: softmol)
  --max-oracle-calls <int>   Max oracle calls (default: 10000)
  --freq-log <int>           AUC logging frequency (default: 100)
  --save-topk <int>          Additional top-k file to save (default: 100)
  --genmol-metrics-csv <p>   Local genmol pmo_metrics csv (default: auto from --genmol-output-dir)
  --genmol-output-dir <dir>  Genmol output dir (default: ../genmol/scripts/exps/pmo/main/genmol/results)
  --dry-run                  Dry run for launcher only
  -h, --help                 Show help
USAGE
}

OUTPUT_DIR="results/pmo/softmol_mcts_seed42_20260327"
SEED=42
GPUS="0,1,2,3,4,5,6,7"
CONDA_ENV="softmol"
MAX_ORACLE_CALLS=10000
FREQ_LOG=100
SAVE_TOPK=100
GENMOL_METRICS_CSV=""
GENMOL_OUTPUT_DIR="../genmol/scripts/exps/pmo/main/genmol/results"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --max-oracle-calls)
      MAX_ORACLE_CALLS="$2"
      shift 2
      ;;
    --freq-log)
      FREQ_LOG="$2"
      shift 2
      ;;
    --save-topk)
      SAVE_TOPK="$2"
      shift 2
      ;;
    --genmol-metrics-csv)
      GENMOL_METRICS_CSV="$2"
      shift 2
      ;;
    --genmol-output-dir)
      GENMOL_OUTPUT_DIR="$2"
      shift 2
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
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

LAUNCH_CMD=(
  bash tool/run_pmo_seed42_8gpu.sh
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
  --gpus "$GPUS"
  --conda-env "$CONDA_ENV"
  --max-oracle-calls "$MAX_ORACLE_CALLS"
  --freq-log "$FREQ_LOG"
  --save-topk "$SAVE_TOPK"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  LAUNCH_CMD+=(--dry-run)
fi

echo "[STEP 1/3] Launch PMO jobs"
"${LAUNCH_CMD[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY-RUN] Skip postprocess in dry-run mode."
  exit 0
fi

echo "[STEP 2/3] Evaluate SoftMol PMO outputs (Genmol-aligned schema)"
conda run -n "$CONDA_ENV" python tool/eval_pmo_batch_align.py \
  --input_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --freq_log "$FREQ_LOG" \
  --max_oracle_calls "$MAX_ORACLE_CALLS"

if [[ -z "$GENMOL_METRICS_CSV" ]]; then
  GENMOL_METRICS_CSV="${GENMOL_OUTPUT_DIR}/pmo_metrics_seed${SEED}.csv"
fi

echo "[STEP 3/3] Compare SoftMol vs local Genmol PMO metrics"
conda run -n "$CONDA_ENV" python tool/compare_pmo_vs_genmol_local.py \
  --seed "$SEED" \
  --softmol_input_dir "$OUTPUT_DIR" \
  --genmol_metrics_csv "$GENMOL_METRICS_CSV"

echo "[DONE] Pipeline finished."
