#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run de novo grid sampling for multiple checkpoints in parallel across GPUs.

Usage:
  bash tool/run_denovo_grid_8gpu.sh [options]

Options:
  --gpus <csv>            GPU ids, default: 0,1,2,3,4,5,6,7
  --p-values <csv>        Nucleus p values, default: 1.0,0.95,0.90,0.85,0.80
  --temp-values <csv>     Temperature values, default: 0.9,1.0,1.1,1.2,1.3
  --repeat <int>          Number of runs per setting, default: 3
  --num-samples <int>     Number of molecules per run, default: 1000
  --eval-bsz <int>        Eval batch size in sample.py, default: 1000
  --seed <int>            Base seed in sample.py, default: 42
  --model <name>          Model config name, default: small-89M
  --steps <int>           Sampling steps (-T), default: 300
  --length <int>          Sequence length (-l), default: 72
  --block-size <int>      Block size (-b), default: 2
  --ckpts <csv>           Checkpoint list relative to repo root
                          default: weights/0-15000-v1.ckpt,weights/89M-epoch6-best.ckpt,weights/best.ckpt
  --out-root <path>       Output root directory, default: results/denovo/grid_<timestamp>
  --env <name>            Conda env name, default: softmol
  -h, --help              Show help
EOF
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf "%s" "$s"
}

parse_pm() {
  local raw
  raw="$(trim "$1")"
  local mean std
  mean="$(trim "${raw%%±*}")"
  if [[ "$raw" == *"±"* ]]; then
    std="$(trim "${raw#*±}")"
  else
    std="0.0"
  fi
  printf "%s,%s" "$mean" "$std"
}

extract_metrics_to_csv_line() {
  local log_file="$1"
  local status="$2"

  local validity_mean="NA" validity_std="NA"
  local uniqueness_mean="NA" uniqueness_std="NA"
  local quality_mean="NA" quality_std="NA"
  local quality2_mean="NA" quality2_std="NA"
  local diversity_mean="NA" diversity_std="NA"
  local time_mean="NA" time_std="NA"

  if [[ "$status" -eq 0 ]]; then
    local summary_line
    summary_line="$(grep -E "±" "$log_file" | tail -n1 || true)"
    if [[ -n "$summary_line" ]]; then
      local c1 c2 c3 c4 c5 c6
      IFS='|' read -r c1 c2 c3 c4 c5 c6 <<< "$summary_line"
      IFS=',' read -r validity_mean validity_std <<< "$(parse_pm "$c1")"
      IFS=',' read -r uniqueness_mean uniqueness_std <<< "$(parse_pm "$c2")"
      IFS=',' read -r quality_mean quality_std <<< "$(parse_pm "$c3")"
      IFS=',' read -r quality2_mean quality2_std <<< "$(parse_pm "$c4")"
      IFS=',' read -r diversity_mean diversity_std <<< "$(parse_pm "$c5")"
      IFS=',' read -r time_mean time_std <<< "$(parse_pm "$c6")"
    else
      local run_line
      run_line="$(grep -E "\\[Run [0-9]+/[0-9]+" "$log_file" | tail -n1 || true)"
      if [[ -n "$run_line" ]]; then
        validity_mean="$(echo "$run_line" | grep -oP 'Validity=\K[0-9]+(?:\.[0-9]+)?' || true)"
        uniqueness_mean="$(echo "$run_line" | grep -oP 'Uniqueness=\K[0-9]+(?:\.[0-9]+)?' || true)"
        quality_mean="$(echo "$run_line" | grep -oP 'Quality=\K[0-9]+(?:\.[0-9]+)?' || true)"
        quality2_mean="$(echo "$run_line" | grep -oP 'Quality2=\K[0-9]+(?:\.[0-9]+)?' || true)"
        diversity_mean="$(echo "$run_line" | grep -oP 'Diversity=\K[0-9]+(?:\.[0-9]+)?' || true)"
        time_mean="$(echo "$run_line" | grep -oP 'Time=\K[0-9]+(?:\.[0-9]+)?(?=s)' || true)"
        # sample.py single-run metrics are in [0, 1]; convert to percentage fields.
        if [[ -n "$validity_mean" ]]; then validity_mean="$(awk -v x="$validity_mean" 'BEGIN{printf "%.6f", x*100.0}')"; fi
        if [[ -n "$uniqueness_mean" ]]; then uniqueness_mean="$(awk -v x="$uniqueness_mean" 'BEGIN{printf "%.6f", x*100.0}')"; fi
        if [[ -n "$quality_mean" ]]; then quality_mean="$(awk -v x="$quality_mean" 'BEGIN{printf "%.6f", x*100.0}')"; fi
        if [[ -n "$quality2_mean" ]]; then quality2_mean="$(awk -v x="$quality2_mean" 'BEGIN{printf "%.6f", x*100.0}')"; fi
        validity_std="0.0"
        uniqueness_std="0.0"
        quality_std="0.0"
        quality2_std="0.0"
        diversity_std="0.0"
        time_std="0.0"
      fi
    fi
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" \
    "$validity_mean" "$validity_std" \
    "$uniqueness_mean" "$uniqueness_std" \
    "$quality_mean" "$quality_std" \
    "$quality2_mean" "$quality2_std" \
    "$diversity_mean" "$diversity_std" \
    "$time_mean" "$time_std"
}

GPUS="0,1,2,3,4,5,6,7"
P_VALUES="1.0,0.95,0.90,0.85,0.80"
TEMP_VALUES="0.9,1.0,1.1,1.2,1.3"
REPEAT=3
NUM_SAMPLES=1000
EVAL_BSZ=1000
SEED=42
MODEL="small-89M"
STEPS=300
LENGTH=72
BLOCK_SIZE=2
CKPTS="weights/0-15000-v1.ckpt,weights/89M-epoch6-best.ckpt,weights/best.ckpt"
OUT_ROOT=""
CONDA_ENV="softmol"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --p-values) P_VALUES="$2"; shift 2 ;;
    --temp-values) TEMP_VALUES="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --eval-bsz) EVAL_BSZ="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --length) LENGTH="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE="$2"; shift 2 ;;
    --ckpts) CKPTS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="results/denovo/grid_$(date +%Y%m%d_%H%M%S)"
fi

LOG_DIR="$OUT_ROOT/logs"
SAMPLE_DIR="$OUT_ROOT/samples"
mkdir -p "$LOG_DIR" "$SAMPLE_DIR"

RESULT_CSV="$OUT_ROOT/grid_metrics.csv"
SUMMARY_MD="$OUT_ROOT/grid_metrics.md"
LOCK_FILE="$OUT_ROOT/.csv.lock"

echo "ckpt,p,temperature,gpu,repeat,num_samples,eval_bsz,seed,status,validity_mean_pct,validity_std_pct,uniqueness_mean_pct,uniqueness_std_pct,quality_mean_pct,quality_std_pct,quality2_mean_pct,quality2_std_pct,diversity_mean,diversity_std,time_mean_s,time_std_s,log_file,sample_out_prefix" > "$RESULT_CSV"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a P_ARR <<< "$P_VALUES"
IFS=',' read -r -a T_ARR <<< "$TEMP_VALUES"
IFS=',' read -r -a CKPT_ARR <<< "$CKPTS"

if [[ "${#GPU_ARR[@]}" -eq 0 ]]; then
  echo "No GPUs parsed from --gpus: $GPUS" >&2
  exit 1
fi

for idx in "${!GPU_ARR[@]}"; do
  GPU_ARR[$idx]="$(trim "${GPU_ARR[$idx]}")"
done
for idx in "${!P_ARR[@]}"; do
  P_ARR[$idx]="$(trim "${P_ARR[$idx]}")"
done
for idx in "${!T_ARR[@]}"; do
  T_ARR[$idx]="$(trim "${T_ARR[$idx]}")"
done
for idx in "${!CKPT_ARR[@]}"; do
  CKPT_ARR[$idx]="$(trim "${CKPT_ARR[$idx]}")"
done

JOBS_FILE="$OUT_ROOT/jobs_all.list"
: > "$JOBS_FILE"

for ckpt in "${CKPT_ARR[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Checkpoint not found: $ckpt" >&2
    exit 1
  fi
  for p in "${P_ARR[@]}"; do
    for t in "${T_ARR[@]}"; do
      echo "${ckpt}|${p}|${t}" >> "$JOBS_FILE"
    done
  done
done

TOTAL_JOBS="$(wc -l < "$JOBS_FILE")"
echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Output root: $OUT_ROOT"
echo "[INFO] Total jobs: $TOTAL_JOBS (${#CKPT_ARR[@]} ckpts x ${#P_ARR[@]} p-values x ${#T_ARR[@]} temperatures)"
echo "[INFO] GPUs: ${GPUS}"
echo "[INFO] repeat=$REPEAT, num_samples=$NUM_SAMPLES, eval_bsz=$EVAL_BSZ"

for gpu in "${GPU_ARR[@]}"; do
  : > "$OUT_ROOT/jobs_gpu${gpu}.list"
done

job_idx=0
while IFS= read -r job; do
  gpu="${GPU_ARR[$((job_idx % ${#GPU_ARR[@]}))]}"
  echo "$job" >> "$OUT_ROOT/jobs_gpu${gpu}.list"
  job_idx=$((job_idx + 1))
done < "$JOBS_FILE"

run_one_job() {
  local gpu="$1"
  local ckpt="$2"
  local p="$3"
  local t="$4"

  local ckpt_base ckpt_name p_tag t_tag tag log_file out_prefix status metrics_csv status_text
  ckpt_base="$(basename "$ckpt")"
  ckpt_name="${ckpt_base%.ckpt}"
  p_tag="${p/./p}"
  t_tag="${t/./p}"
  tag="${ckpt_name}_p${p_tag}_t${t_tag}"
  log_file="$LOG_DIR/${tag}.log"
  out_prefix="$SAMPLE_DIR/${tag}"

  echo "[GPU ${gpu}] START $tag"
  if conda run -n "$CONDA_ENV" python -u sample.py \
      -g "$gpu" \
      -l "$LENGTH" \
      -b "$BLOCK_SIZE" \
      -c "$ckpt" \
      -m "$MODEL" \
      -T "$STEPS" \
      -p "$p" \
      --temperature "$t" \
      -e "$EVAL_BSZ" \
      -n "$NUM_SAMPLES" \
      -r "$REPEAT" \
      -s "$SEED" \
      -o "$out_prefix" > "$log_file" 2>&1; then
    status=0
    status_text="ok"
  else
    status=$?
    status_text="fail($status)"
  fi

  metrics_csv="$(extract_metrics_to_csv_line "$log_file" "$status")"
  {
    flock 200
    echo "${ckpt},${p},${t},${gpu},${REPEAT},${NUM_SAMPLES},${EVAL_BSZ},${SEED},${status_text},${metrics_csv},${log_file},${out_prefix}" >> "$RESULT_CSV"
  } 200>"$LOCK_FILE"
  echo "[GPU ${gpu}] DONE  $tag status=${status_text}"
}

PIDS=()
for gpu in "${GPU_ARR[@]}"; do
  job_file="$OUT_ROOT/jobs_gpu${gpu}.list"
  if [[ ! -s "$job_file" ]]; then
    continue
  fi
  (
    while IFS='|' read -r ckpt p t; do
      run_one_job "$gpu" "$ckpt" "$p" "$t"
    done < "$job_file"
  ) &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

rm -f "$LOCK_FILE"

{
  echo "| ckpt | p | temperature | validity (%) | uniqueness (%) | quality (%) | quality2 (%) | diversity | time (s) | status |"
  echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
  tail -n +2 "$RESULT_CSV" | sort -t',' -k1,1 -k2,2n -k3,3n | \
    awk -F',' '{printf "| %s | %s | %s | %s ± %s | %s ± %s | %s ± %s | %s ± %s | %s ± %s | %s ± %s | %s |\n", $1,$2,$3,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$9}'
} > "$SUMMARY_MD"

echo "[INFO] Done."
echo "[INFO] CSV: $RESULT_CSV"
echo "[INFO] MD : $SUMMARY_MD"
