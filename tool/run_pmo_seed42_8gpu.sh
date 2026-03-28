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
  --procs-per-gpu <int>      Concurrent processes per GPU (default: 1)
  --conda-env <name>         Conda env name (default: softmol)
  --max-oracle-calls <int>   Max oracle calls (default: 10000)
  --freq-log <int>           AUC logging frequency (default: 100)
  --save-topk <int>          Additional top-k file to save (default: 100)
  --ckpt <path>              Model checkpoint (default: weights/89M-epoch6-best.ckpt)
  --length <int>             Sequence length (default: 512)
  --block-size <int>         Block size (default: 8)
  --nucleus <float>          Nucleus sampling threshold (default: 1.0)
  --temperature <float>      Sampling temperature (default: 1.1)
  --value-weight <float>     Value weight (default: 0.0)
  --search-time <int>        Search iteration upper bound (default: 100000)
  --min-terminals <int>      Minimum terminal nodes (default: -1)
  --max-split-depth <int>    Max split depth (default: 100)
  --init-children <int>      Initial children for root node (default: 20)
  --n-total-children <int>   Children for non-root nodes (default: 8)
  --c-param <float>          UCB exploration coefficient (default: 2.1)
  --width-increase-factor <int>  Adaptive width increase factor (default: 2)
  --add-value-weight <float> Additional value weight (default: 0.0)
  --n-simulations <int>      Number of simulations (default: 1)
  --fastrollout-weight <float>   Fast rollout weight (default: 1.0)
  --greedy-path              Enable greedy path (default: disabled)
  --trace-dir <dir>          Optional trace directory for per-task traces
  --max-n-repeat <int>       Max repeat limit for same path (default: 5)
  --retry-once               Retry failed jobs once (default: enabled)
  --no-retry                 Disable retry
  --dry-run                  Print commands only
  -h, --help                 Show this message

Notes:
  - 23 PMO tasks, single seed.
  - Wave scheduling with total concurrency = (#GPUs * --procs-per-gpu).
  - First-round log file: <oracle>_seed<seed>.log
  - Retry-round log file: <oracle>_seed<seed>_retry1.log
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="results/pmo/softmol_mcts_seed42_20260327"
SEED=42
GPU_IDS="0,1,2,3,4,5,6,7"
PROCS_PER_GPU=1
CONDA_ENV="softmol"
MAX_ORACLE_CALLS=10000
FREQ_LOG=100
SAVE_TOPK=100
CKPT="weights/89M-epoch6-best.ckpt"
VOCAB="vocab_V2.txt"
LENGTH=100
BLOCK_SIZE=2
STEPS=128
NUCLEUS=1.0
TEMPERATURE=1.1
GEN_BATCH_SIZE=1
MODEL_NAME="small-89M"
VALUE_WEIGHT=0.0
SEARCH_TIME=100000
MIN_TERMINALS=-1
MAX_SPLIT_DEPTH=100
INIT_CHILDREN=20
N_TOTAL_CHILDREN=8
C_PARAM=2.1
WIDTH_INCREASE_FACTOR=2
ADD_VALUE_WEIGHT=0.0
N_SIMULATIONS=1
FASTROLLOUT_WEIGHT=1.0
GREEDY_PATH=0
TRACE_DIR=""
MAX_N_REPEAT=5
DIVERSITY_THRESHOLD=0.6
MAX_RESAMPLE_ON_EMPTY=5
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
    --procs-per-gpu)
      PROCS_PER_GPU="$2"
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
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --length)
      LENGTH="$2"
      shift 2
      ;;
    --block-size)
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --nucleus)
      NUCLEUS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --value-weight)
      VALUE_WEIGHT="$2"
      shift 2
      ;;
    --search-time)
      SEARCH_TIME="$2"
      shift 2
      ;;
    --min-terminals)
      MIN_TERMINALS="$2"
      shift 2
      ;;
    --max-split-depth)
      MAX_SPLIT_DEPTH="$2"
      shift 2
      ;;
    --init-children)
      INIT_CHILDREN="$2"
      shift 2
      ;;
    --n-total-children)
      N_TOTAL_CHILDREN="$2"
      shift 2
      ;;
    --c-param)
      C_PARAM="$2"
      shift 2
      ;;
    --width-increase-factor)
      WIDTH_INCREASE_FACTOR="$2"
      shift 2
      ;;
    --add-value-weight)
      ADD_VALUE_WEIGHT="$2"
      shift 2
      ;;
    --n-simulations)
      N_SIMULATIONS="$2"
      shift 2
      ;;
    --fastrollout-weight)
      FASTROLLOUT_WEIGHT="$2"
      shift 2
      ;;
    --greedy-path)
      GREEDY_PATH=1
      shift 1
      ;;
    --trace-dir)
      TRACE_DIR="$2"
      shift 2
      ;;
    --max-n-repeat)
      MAX_N_REPEAT="$2"
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

if ! [[ "$PROCS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --procs-per-gpu must be a positive integer, got: $PROCS_PER_GPU" >&2
  exit 1
fi

SLOT_GPU_ARR=()
for gpu in "${GPU_ARR[@]}"; do
  for ((slot_i=0; slot_i<PROCS_PER_GPU; slot_i++)); do
    SLOT_GPU_ARR+=("$gpu")
  done
done

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

  local num_slots=${#SLOT_GPU_ARR[@]}
  local wave_count=0

  for ((i=0; i<${#jobs[@]}; i+=num_slots)); do
    wave_count=$((wave_count + 1))
    pids=()
    orcs=()
    gpus=()

    for ((j=0; j<num_slots; j++)); do
      idx=$((i + j))
      if [[ $idx -ge ${#jobs[@]} ]]; then
        break
      fi

      oracle="${jobs[$idx]}"
      gpu="${SLOT_GPU_ARR[$j]}"

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
        --save_topk "$SAVE_TOPK"
        --ckpt "$CKPT"
        --vocab "$VOCAB"
        --length "$LENGTH"
        --block_size "$BLOCK_SIZE"
        --steps "$STEPS"
        --nucleus "$NUCLEUS"
        --temperature "$TEMPERATURE"
        --gen_batch_size "$GEN_BATCH_SIZE"
        --model "$MODEL_NAME"
        --value_weight "$VALUE_WEIGHT"
        --search_time "$SEARCH_TIME"
        --min_terminals "$MIN_TERMINALS"
        --max_split_depth "$MAX_SPLIT_DEPTH"
        --init_children "$INIT_CHILDREN"
        --n_total_children "$N_TOTAL_CHILDREN"
        --c_param "$C_PARAM"
        --width_increase_factor "$WIDTH_INCREASE_FACTOR"
        --add_value_weight "$ADD_VALUE_WEIGHT"
        --n_simulations "$N_SIMULATIONS"
        --fastrollout_weight "$FASTROLLOUT_WEIGHT"
        --max_n_repeat "$MAX_N_REPEAT"
        --diversity_threshold "$DIVERSITY_THRESHOLD"
        --max_resample_on_empty "$MAX_RESAMPLE_ON_EMPTY"
      )
      if [[ "$GREEDY_PATH" -eq 1 ]]; then
        cmd+=(--greedy_path)
      fi
      if [[ -n "$TRACE_DIR" ]]; then
        cmd+=(--trace_path "$TRACE_DIR")
      fi

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
echo "[INFO] Procs/GPU:    $PROCS_PER_GPU"
echo "[INFO] Max parallel: ${#SLOT_GPU_ARR[@]}"
echo "[INFO] Oracle count: ${#ORACLES[@]}"
echo "[INFO] Save top-k:   $SAVE_TOPK (plus always top10/top100)"
echo "[INFO] Model ckpt:   $CKPT"
echo "[INFO] Sampler:      length=$LENGTH block_size=$BLOCK_SIZE p=$NUCLEUS T=$TEMPERATURE"
echo "[INFO] MCTS:         search_time=$SEARCH_TIME init_children=$INIT_CHILDREN n_total_children=$N_TOTAL_CHILDREN c_param=$C_PARAM"
echo "[INFO] MCTS extra:   value_weight=$VALUE_WEIGHT min_terminals=$MIN_TERMINALS max_split_depth=$MAX_SPLIT_DEPTH width_increase_factor=$WIDTH_INCREASE_FACTOR add_value_weight=$ADD_VALUE_WEIGHT n_simulations=$N_SIMULATIONS fastrollout_weight=$FASTROLLOUT_WEIGHT greedy_path=$GREEDY_PATH max_n_repeat=$MAX_N_REPEAT"
echo "[INFO] Trace dir:    ${TRACE_DIR:-<disabled>}"
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
