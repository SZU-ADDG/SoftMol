#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
用法：
  bash evaluate_checkpoints_auto_default_gpu.sh <CKPT_DIR> [OUTPUT_CSV] [选项]

必选参数：
  <CKPT_DIR>          包含 .ckpt 文件的目录（递归查找）

可选位置参数：
  [OUTPUT_CSV]        输出 CSV 路径（默认: results_default_gpu.csv）

可选参数：
  -g, --gpus          GPU id 列表，逗号分隔（默认: "0"）
  -h, --help          显示帮助

说明：
  1) 脚本仅向 sample.py 传递 -g 和 -c：
       python -u sample.py -g <single_gpu_id> -c <ckpt_path>
  2) 其它采样参数保持 sample.py 默认值
  3) 多卡并行策略：每张卡 1 个 worker，worker 内串行处理分配到的 ckpt
  4) 默认断点续跑：已在 CSV 中 status=OK 的 ckpt 会跳过
USAGE
}

GPU_IDS="0"
OUTPUT_CSV="results_default_gpu.csv"

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CKPT_DIR="$1"
shift

if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
  OUTPUT_CSV="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "错误: 找不到目录 '$CKPT_DIR'"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SAMPLE_PY="$PROJECT_ROOT/sample.py"

if [[ ! -f "$SAMPLE_PY" ]]; then
  echo "错误: 找不到 sample.py: $SAMPLE_PY"
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY_RAW <<< "$GPU_IDS"
GPU_ARRAY=()
for gpu in "${GPU_ARRAY_RAW[@]}"; do
  gpu_trimmed="$(echo "$gpu" | xargs)"
  if [[ -n "$gpu_trimmed" ]]; then
    GPU_ARRAY+=("$gpu_trimmed")
  fi
done

NUM_GPUS="${#GPU_ARRAY[@]}"
if [[ "$NUM_GPUS" -eq 0 ]]; then
  echo "错误: 未解析到有效 GPU 列表（传入: '$GPU_IDS'）"
  exit 1
fi

CSV_HEADER="ckpt_filename,validity,uniqueness,diversity,quality,quality2,sample_time,status,gpu_id"
if [[ -f "$OUTPUT_CSV" ]]; then
  if [[ ! -s "$OUTPUT_CSV" ]]; then
    echo "$CSV_HEADER" > "$OUTPUT_CSV"
  else
    first_line="$(head -n 1 "$OUTPUT_CSV")"
    if [[ "$first_line" != "$CSV_HEADER" ]]; then
      echo "错误: 现有 CSV 表头不匹配。"
      echo "当前脚本期望表头: $CSV_HEADER"
      echo "请改用新的 OUTPUT_CSV 文件，或备份后重建。"
      exit 1
    fi
  fi
else
  echo "$CSV_HEADER" > "$OUTPUT_CSV"
fi

declare -A DONE_OK=()
while IFS= read -r done_ckpt; do
  [[ -n "$done_ckpt" ]] && DONE_OK["$done_ckpt"]=1
done < <(
  awk -F',' '
    NR > 1 {
      ck = $1
      st = tolower($8)
      gsub(/^"|"$/, "", ck)
      if (st == "ok") print ck
    }
  ' "$OUTPUT_CSV"
)

mapfile -d $'\0' -t CKPTS < <(find "$CKPT_DIR" -type f -name "*.ckpt" -print0 | sort -z)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "提示: 在 '$CKPT_DIR' 下未找到 .ckpt 文件。"
  exit 0
fi

PENDING_CKPTS=()
for ckpt_path in "${CKPTS[@]}"; do
  ckpt_name="$(basename "$ckpt_path")"
  if [[ -n "${DONE_OK[$ckpt_name]:-}" ]]; then
    continue
  fi
  PENDING_CKPTS+=("$ckpt_path")
done

skipped_count=$(( ${#CKPTS[@]} - ${#PENDING_CKPTS[@]} ))
echo "共找到 ${#CKPTS[@]} 个 ckpt，已完成并跳过 ${skipped_count} 个，待处理 ${#PENDING_CKPTS[@]} 个。"
if [[ "${#PENDING_CKPTS[@]}" -eq 0 ]]; then
  echo "没有需要处理的 ckpt。"
  exit 0
fi

is_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

to_decimal() {
  awk -v x="$1" 'BEGIN{printf "%.6f", x/100.0}'
}

parse_or_na() {
  local pattern="$1"
  local file="$2"
  local value
  value="$(grep -oP "$pattern" "$file" | head -n 1 || true)"
  if [[ -z "$value" ]]; then
    echo "NA"
  else
    echo "$value"
  fi
}

parse_metrics() {
  local file="$1"
  VALIDITY="NA"
  UNIQUENESS="NA"
  DIVERSITY="NA"
  QUALITY="NA"
  QUALITY2="NA"
  SAMPLE_TIME="NA"

  local summary_line
  summary_line="$(grep -E "±" "$file" | tail -n 1 || true)"

  if [[ -n "$summary_line" ]]; then
    local m1 m2 m3 m4 m5 m6
    read -r m1 m2 m3 m4 m5 m6 < <(
      printf "%s\n" "$summary_line" | awk -F'\\|' '
        {
          for (i = 1; i <= 6; i++) {
            gsub(/^ +| +$/, "", $i)
            split($i, a, " ± ")
            printf "%s%s", a[1], (i < 6 ? " " : "")
          }
        }
      '
    )
    if is_number "$m1" && is_number "$m2" && is_number "$m3" && is_number "$m4" && is_number "$m5" && is_number "$m6"; then
      VALIDITY="$(to_decimal "$m1")"
      UNIQUENESS="$(to_decimal "$m2")"
      QUALITY="$(to_decimal "$m3")"
      QUALITY2="$(to_decimal "$m4")"
      DIVERSITY="$m5"
      SAMPLE_TIME="$m6"
      return 0
    fi
  fi

  VALIDITY="$(parse_or_na '(?i)Validity=\K[0-9]+(?:\.[0-9]+)?' "$file")"
  UNIQUENESS="$(parse_or_na '(?i)Uniqueness=\K[0-9]+(?:\.[0-9]+)?' "$file")"
  DIVERSITY="$(parse_or_na '(?i)Diversity=\K[0-9]+(?:\.[0-9]+)?' "$file")"
  QUALITY="$(parse_or_na '(?i)Quality=\K[0-9]+(?:\.[0-9]+)?' "$file")"
  QUALITY2="$(parse_or_na '(?i)Quality2=\K[0-9]+(?:\.[0-9]+)?' "$file")"
  SAMPLE_TIME="$(parse_or_na '(?i)Time=\K[0-9]+(?:\.[0-9]+)?(?=s)' "$file")"
}

esc_csv() {
  sed 's/"/""/g' <<< "$1"
}

TMP_DIR="$(mktemp -d --suffix=.eval_default_gpu)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

ASSIGN_FILES=()
WORKER_CSV_FILES=()
for ((i = 0; i < NUM_GPUS; i++)); do
  assign_file="$TMP_DIR/gpu_${i}.list"
  worker_csv="$TMP_DIR/gpu_${i}.csv"
  : > "$assign_file"
  : > "$worker_csv"
  ASSIGN_FILES+=("$assign_file")
  WORKER_CSV_FILES+=("$worker_csv")
done

for ((i = 0; i < ${#PENDING_CKPTS[@]}; i++)); do
  gpu_index=$((i % NUM_GPUS))
  printf '%s\n' "${PENDING_CKPTS[$i]}" >> "${ASSIGN_FILES[$gpu_index]}"
done

PIDS=()
for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
  if [[ ! -s "${ASSIGN_FILES[$gpu_index]}" ]]; then
    continue
  fi
  (
    # 子进程不负责清理共享临时目录，避免提前删除影响其它 GPU worker
    trap - EXIT
    # 参考旧脚本思路：worker 内部不因单条命令失败而整体退出
    set +e
    single_gpu_id="${GPU_ARRAY[$gpu_index]}"
    worker_csv="${WORKER_CSV_FILES[$gpu_index]}"
    while IFS= read -r ckpt_file_path; do
      [[ -z "$ckpt_file_path" ]] && continue
      ckpt_filename="$(basename "$ckpt_file_path")"
      echo "===== GPU ${single_gpu_id} 处理: ${ckpt_filename} ====="

      tmp_out="$(mktemp --tmpdir="$TMP_DIR" "gpu${single_gpu_id}_XXXX.out")"
      if [[ -z "$tmp_out" || ! -f "$tmp_out" ]]; then
        echo "警告: 无法创建临时输出文件，跳过 $ckpt_filename"
        ckpt_esc="$(esc_csv "$ckpt_filename")"
        echo "\"$ckpt_esc\",NA,NA,NA,NA,NA,NA,FAILED,$single_gpu_id" >> "$worker_csv"
        continue
      fi

      (
        cd "$PROJECT_ROOT"
        python -u sample.py -g "$single_gpu_id" -c "$ckpt_file_path"
      ) 2>&1 | tee "$tmp_out"
      # 取 python 的退出码，不受 tee 影响
      status=${PIPESTATUS[0]}

      parse_metrics "$tmp_out"

      if [[ "$status" -eq 0 ]]; then
        status_str="OK"
      else
        status_str="FAILED"
      fi

      ckpt_esc="$(esc_csv "$ckpt_filename")"
      echo "\"$ckpt_esc\",$VALIDITY,$UNIQUENESS,$DIVERSITY,$QUALITY,$QUALITY2,$SAMPLE_TIME,$status_str,$single_gpu_id" >> "$worker_csv"

      if [[ "$status_str" == "OK" ]]; then
        echo "完成: ${ckpt_filename} (GPU ${single_gpu_id})"
      else
        echo "警告: ${ckpt_filename} 失败 (GPU ${single_gpu_id})，已记录为 FAILED。"
      fi

      rm -f "$tmp_out"
    done < "${ASSIGN_FILES[$gpu_index]}"
    exit 0
  ) &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

produced_rows=0
for worker_csv in "${WORKER_CSV_FILES[@]}"; do
  if [[ -s "$worker_csv" ]]; then
    rows_in_file=$(wc -l < "$worker_csv")
    produced_rows=$((produced_rows + rows_in_file))
    cat "$worker_csv" >> "$OUTPUT_CSV"
  fi
done

if [[ "$produced_rows" -lt "${#PENDING_CKPTS[@]}" ]]; then
  missing_rows=$(( ${#PENDING_CKPTS[@]} - produced_rows ))
  echo "警告: 本次应处理 ${#PENDING_CKPTS[@]} 个，但仅写入 ${produced_rows} 行，缺少 ${missing_rows} 行。"
  echo "建议直接重跑同一命令，脚本会基于 status=OK 自动跳过已完成项。"
fi

echo "全部完成：本次处理 ${#PENDING_CKPTS[@]} 个 ckpt。结果已追加到: $OUTPUT_CSV"
