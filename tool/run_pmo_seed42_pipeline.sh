#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash tool/run_pmo_seed42_pipeline.sh [options]

Options:
  --output-dir <dir>   PMO output dir (default: results/pmo/softmol_mcts_seed42_20260327)
  --seed <int>         Seed (default: 42)
  --gpus <csv>         GPUs (default: 0,1,2,3,4,5,6,7)
  --conda-env <name>   Conda env (default: softmol)
  --dry-run            Dry run for launcher only
  -h, --help           Show help
USAGE
}

OUTPUT_DIR="results/pmo/softmol_mcts_seed42_20260327"
SEED=42
GPUS="0,1,2,3,4,5,6,7"
CONDA_ENV="softmol"
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
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  LAUNCH_CMD+=(--dry-run)
fi

echo "[STEP 1/2] Launch PMO jobs"
"${LAUNCH_CMD[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DRY-RUN] Skip postprocess in dry-run mode."
  exit 0
fi

echo "[STEP 2/2] Summarize + validate + compare"
conda run -n "$CONDA_ENV" python tool/compare_pmo_seed42_vs_table3.py \
  --input_dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --freq_log 100 \
  --max_oracle_calls 10000

echo "[DONE] Pipeline finished."
