#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash tool/run_pmo_seed42_8gpu.sh [options]

Options:
  --output-dir <dir>         Output directory (default: results/pmo/softmol_mcts_seed42_20260327)
  --seed <int>               Seed (default: 42)
  --gpus <csv>               GPU ids, comma-separated (default: 0,1,2,3,4,5,6,7)
  --conda-env <name>         Conda env name (default: softmol)
  --max-oracle-calls <int>   Max oracle calls (default: 10000)
  --freq-log <int>           AUC logging frequency (default: 100)
  --retry-once               Retry failed jobs once (default: enabled)
  --no-retry                 Disable retry
  --dry-run                  Print commands only
  -h, --help                 Show this message

Notes:
  - 23 PMO tasks, single seed.
  - One process per GPU, wave scheduling.
  - First-round log file: <oracle>_seed<seed>.log
  - Retry-round log file: <oracle>_seed<seed>_retry1.log
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="results/pmo/softmol_mcts_seed42_20260327"
SEED=42
GPU_IDS="0,1,2,3,4,5,6,7"
CONDA_ENV="softmol"
MAX_ORACLE_CALLS=10000
FREQ_LOG=100
RETRY_ONCE=1
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
      GPU_IDS="$2"
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
    --retry-once)
      RETRY_ONCE=1
      shift 1
      ;;
    --no-retry)
      RETRY_ONCE=0
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
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$PROJECT_ROOT"

IFS=',' read -r -a GPU_ARR_RAW <<< "$GPU_IDS"
GPU_ARR=()
for g in "${GPU_ARR_RAW[@]}"; do
  g_trimmed="$(echo "$g" | xargs)"
  if [[ -n "$g_trimmed" ]]; then
    GPU_ARR+=("$g_trimmed")
  fi
done

if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "[ERROR] No valid GPU IDs parsed from: $GPU_IDS" >&2
  exit 1
fi

ORACLES=(
  albuterol_similarity
  amlodipine_mpo
  celecoxib_rediscovery
  deco_hop
  drd2
  fexofenadine_mpo
  gsk3b
  isomers_c7h8n2o2
  isomers_c9h10n2o2pf2cl
  jnk3
  median1
  median2
  mestranol_similarity
  osimertinib_mpo
  perindopril_mpo
  qed
  ranolazine_mpo
  scaffold_hop
  sitagliptin_mpo
  thiothixene_rediscovery
  troglitazone_rediscovery
  valsartan_smarts
  zaleplon_mpo
)

OUT_ABS="$PROJECT_ROOT/$OUTPUT_DIR"
LOG_DIR="$OUT_ABS/logs"
mkdir -p "$LOG_DIR"

FAILED_FILE="$OUT_ABS/failed_jobs.txt"
FAILED_RETRY_FILE="$OUT_ABS/failed_jobs_retry1.txt"
: > "$FAILED_FILE"
: > "$FAILED_RETRY_FILE"

run_one_round() {
  local round_name="$1"
  local failed_out="$2"
  shift 2
  local jobs=("$@")

  local num_gpus=${#GPU_ARR[@]}
  local wave_count=0

  for ((i=0; i<${#jobs[@]}; i+=num_gpus)); do
    wave_count=$((wave_count + 1))
    pids=()
    orcs=()
    gpus=()

    for ((j=0; j<num_gpus; j++)); do
      idx=$((i + j))
      if [[ $idx -ge ${#jobs[@]} ]]; then
        break
      fi

      oracle="${jobs[$idx]}"
      gpu="${GPU_ARR[$j]}"

      if [[ "$round_name" == "round1" ]]; then
        log_path="$LOG_DIR/${oracle}_seed${SEED}.log"
      else
        log_path="$LOG_DIR/${oracle}_seed${SEED}_${round_name}.log"
      fi

      cmd=(
        conda run -n "$CONDA_ENV" python gated_mcts/run_pmo_mcts.py
        --oracle_name "$oracle"
        --seed "$SEED"
        --device "$gpu"
        --output_dir "$OUTPUT_DIR"
        --max_oracle_calls "$MAX_ORACLE_CALLS"
        --freq_log "$FREQ_LOG"
        --ckpt "weights/89M-epoch6-best.ckpt"
        --vocab "vocab_V2.txt"
        --length 512
        --block_size 8
        --steps 128
        --nucleus 1.0
        --temperature 1.1
        --gen_batch_size 1
        --model "small-89M"
        --search_time 100000
        --init_children 20
        --n_total_children 8
        --c_param 2.1
      )

      echo "[${round_name}] [WAVE-${wave_count}] launch oracle=${oracle} gpu=${gpu} log=${log_path}"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  CMD: ${cmd[*]}"
      else
        "${cmd[@]}" >"$log_path" 2>&1 &
        pids+=("$!")
        orcs+=("$oracle")
        gpus+=("$gpu")
      fi
    done

    if [[ "$DRY_RUN" -eq 1 ]]; then
      continue
    fi

    for k in "${!pids[@]}"; do
      pid="${pids[$k]}"
      oracle="${orcs[$k]}"
      gpu="${gpus[$k]}"
      if wait "$pid"; then
        echo "[${round_name}] [OK] oracle=${oracle} gpu=${gpu}"
      else
        echo "[${round_name}] [FAILED] oracle=${oracle} gpu=${gpu}" >&2
        echo "$oracle" >> "$failed_out"
      fi
    done
  done
}

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Output dir:   $OUTPUT_DIR"
echo "[INFO] Seed:         $SEED"
echo "[INFO] GPUs:         ${GPU_ARR[*]}"
echo "[INFO] Oracle count: ${#ORACLES[@]}"
echo "[INFO] Retry once:   $RETRY_ONCE"

run_one_round "round1" "$FAILED_FILE" "${ORACLES[@]}"

if [[ "$DRY_RUN" -eq 0 ]]; then
  failed_n=$(wc -l < "$FAILED_FILE" | xargs)
  echo "[INFO] Round1 failed jobs: $failed_n"

  if [[ "$RETRY_ONCE" -eq 1 && "$failed_n" -gt 0 ]]; then
    mapfile -t failed_jobs < "$FAILED_FILE"
    echo "[INFO] Start retry round for failed jobs..."
    run_one_round "retry1" "$FAILED_RETRY_FILE" "${failed_jobs[@]}"
    failed_retry_n=$(wc -l < "$FAILED_RETRY_FILE" | xargs)
    echo "[INFO] Retry1 failed jobs: $failed_retry_n"
  fi

  echo "[INFO] Done. Logs: $LOG_DIR"
  echo "[INFO] Failed list (round1): $FAILED_FILE"
  echo "[INFO] Failed list (retry1): $FAILED_RETRY_FILE"
else
  echo "[DRY-RUN] Finished without execution."
fi
