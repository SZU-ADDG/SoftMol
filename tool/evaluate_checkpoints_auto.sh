#!/usr/bin/env bash

#
# 批量评估脚本：
# - 接收文件夹路径，自动查找其中的 .ckpt
# - 对每个 .ckpt 调用 sample.py（SMILES 模式，独立采样与评估）
# - 从 sample.py 的输出中解析 Validity / Uniqueness / Diversity / Quality / Quality2 / Time
# - 将结果以追加方式写入 CSV（首行含表头）
#
# 用法：
# bash evaluate_checkpoints_auto.sh outputs/DrugLikeSMILSE-12B-427M-filterLen72/bd3lm-small-89M-len72-bs8  evaluate_checkpoints_resulet/DrugLikeSMILSE-12B-427M-filterLen72-bd3lm-small-89M-len72-bs8-sampleBs8.csv -m small-89M -v vocab_V2.txt  -g "0,1" -l 512 -b 8 -r 3


set -euo pipefail

# -------- 帮助信息 --------
usage() {
  cat <<'USAGE'
用法：
  bash $(basename "$0") <CKPT_DIR> [OUTPUT_CSV] [选项]

必选参数：
  <CKPT_DIR>          包含 .ckpt 文件的文件夹路径

可选位置参数：
  [OUTPUT_CSV]        输出 CSV 文件的路径 (默认: results.csv)

采样控制选项：
  -g, --gpus          GPU id 列表 (默认: "0,1,2,3,4,5,6,7")
  -l, --length        序列上限 (默认: 512)
  -b, --block-size    块大小 (默认: 4)
  -T, --steps         块内步数 (默认: 128)
  -p, --nucleus       Top-p 截断 (默认: 0.95)
  -r, --repeat        重复运行次数 (默认: 1)
  -m, --model         模型规模 (默认: small-89M)
  -e, --eval-bsz      评估批量大小 (默认: 1000)
  -o, --out-dir       采样日志目录 (默认: /share/home/.../evaluate_checkpoints)
  -s, --seed          采样随机种子 (默认: 42)
  -v, --vocab         词表路径 (默认: vocab.txt)
  --prefix            前缀 SMILES（可选）
  --next-block-only   仅生成前缀后的下一块（可选）
  -h, --help          显示此帮助

示例：
  bash $(basename "$0") /path/to/checkpoints results.csv -p 0.9 -r 3
USAGE
}

# -------- 参数与默认值 --------
GPU_IDS="0,1,2,3,4,5,6,7"
L_VALUE=512
B_VALUE=4
T_VALUE=128
P_VALUE=0.95
R_VALUE=1
M_VALUE="small-89M"
E_VALUE=1000
SEED_VALUE=42
SAMPLE_LOG_DIR="/share/home/tm866079609100000/a875465180/yqw_bd3lms/bd3lms-1.0/sample_logs/evaluate_checkpoints"
OUTPUT_CSV="results.csv"
VOCAB_PATH="vocab.txt"
PREFIX=""
NEXT_BLOCK_ONLY=false

# -------- 解析位置参数 --------
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
CKPT_DIR="$1"
shift
if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
  OUTPUT_CSV="$1"
  shift
fi

# -------- 解析选项参数 --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpus) GPU_IDS="$2"; shift 2;;
    -l|--length) L_VALUE="$2"; shift 2;;
    -b|--block-size) B_VALUE="$2"; shift 2;;
    -T|--steps) T_VALUE="$2"; shift 2;;
    -p|--nucleus) P_VALUE="$2"; shift 2;;
    -r|--repeat) R_VALUE="$2"; shift 2;;
    -m|--model) M_VALUE="$2"; shift 2;;
    -e|--eval-bsz) E_VALUE="$2"; shift 2;;
    -o|--out-dir) SAMPLE_LOG_DIR="$2"; shift 2;;
    -s|--seed) SEED_VALUE="$2"; shift 2;;
    -v|--vocab) VOCAB_PATH="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --next-block-only) NEXT_BLOCK_ONLY=true; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数: $1"; usage; exit 1;;
  esac
done

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "错误: 找不到目录 '$CKPT_DIR'"
    exit 1
fi

# -------- 准备 CSV 表头 --------
if [[ ! -f "$OUTPUT_CSV" ]]; then
  echo "ckpt_filename,Validity,Uniqueness,Diversity,Quality,Quality2,Sample_time(seconds),total_generations,p_value,l_value,b_value,T_value,n_value,m_value,e_value,gpu_count,seed" > "$OUTPUT_CSV"
fi

# 创建临时列表文件用于收集临时CSV
TMP_LIST=$(mktemp --suffix=.list)
echo "临时列表文件: $TMP_LIST"

# -------- 遍历并处理 ckpt --------

# 使用 -print0 以安全处理包含空格的路径
mapfile -d $'\0' -t CKPTS < <(find "$CKPT_DIR" -type f -name "*.ckpt" -print0 | sort -z)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "提示：在 '$CKPT_DIR' 下未找到 .ckpt 文件。"
  exit 0
fi

echo "在 '$CKPT_DIR' 下共找到 ${#CKPTS[@]} 个 .ckpt 文件。"

# 分配 ckpt 到 GPU（基于用户传入的 GPU 列表）
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}
if [[ $NUM_GPUS -eq 0 ]]; then
  echo "错误：未解析到可用 GPU ID（传入: '$GPU_IDS'）"; exit 1
fi
CKPT_PER_GPU=$(( (${#CKPTS[@]} + NUM_GPUS - 1) / NUM_GPUS ))  # 向上取整
GPU_ASSIGNMENTS=()
for ((i=0; i<NUM_GPUS; i++)); do
  GPU_ASSIGNMENTS[$i]=""
done

for ((i=0; i<${#CKPTS[@]}; i++)); do
  GPU_INDEX=$((i % NUM_GPUS))
  GPU_ASSIGNMENTS[$GPU_INDEX]="${GPU_ASSIGNMENTS[$GPU_INDEX]} ${CKPTS[$i]}"
done

# 并行运行每个 GPU 的任务
PIDS=()
for ((GPU_ID=0; GPU_ID<NUM_GPUS; GPU_ID++)); do
  if [[ -z "${GPU_ASSIGNMENTS[$GPU_ID]}" ]]; then
    continue
  fi
  (
    for CKPT_FILE_PATH in ${GPU_ASSIGNMENTS[$GPU_ID]}; do
      CKPT_FILENAME=$(basename "$CKPT_FILE_PATH")
      echo "===== GPU $GPU_ID 处理检查点：$CKPT_FILENAME ====="

      # 每个 ckpt 使用单个 GPU（按传入 GPU 列表映射）
      SINGLE_GPU_ID="${GPU_ARRAY[$GPU_ID]}"
      GPU_COUNT=1
      TOTAL_GENERATIONS=$((R_VALUE * E_VALUE)) # 近似总量 = 重复次数 * 每次评估批量大小

      # 将命令输出同时保存到临时文件并打印到终端
      TMP_OUT=$(mktemp)
      set +e
      python -u sample.py \
        -g "$SINGLE_GPU_ID" \
        -l "$L_VALUE" \
        -b "$B_VALUE" \
        -c "$CKPT_FILE_PATH" \
        -T "$T_VALUE" \
        -p "$P_VALUE" \
        -r "$R_VALUE" \
        -m "$M_VALUE" \
        -o "$SAMPLE_LOG_DIR" \
        -e "$E_VALUE" \
        -s "$SEED_VALUE" \
        -v "$VOCAB_PATH" \
        ${PREFIX:+--prefix "$PREFIX"} \
        $( $NEXT_BLOCK_ONLY && echo "--next-block-only" ) 2>&1 | tee "$TMP_OUT"
      STATUS=${PIPESTATUS}
      set -e

      # 解析指标（允许缺失时回退为 NA）
      parse_or_na() {
        local pat="$1"; local file="$2"; local val
        val=$(grep -oP "$pat" "$file" | head -n1 || true)
        if [[ -z "$val" ]]; then echo "NA"; else echo "$val"; fi
      }

      # 当 -r>1 时，优先解析 sample.py 的汇总均值行（带“±”），并按需要缩放百分比为小数
      if [[ "$R_VALUE" -gt 1 ]]; then
        SUMMARY_LINE=$(grep -E "±" "$TMP_OUT" | tail -n1 || true)
        if [[ -n "$SUMMARY_LINE" ]]; then
          read -r M1 M2 M3 M4 M5 M6 < <(printf "%s\n" "$SUMMARY_LINE" | awk -F'\|' '{for(i=1;i<=6;i++){gsub(/^ +| +$/, "", $i); split($i, arr, " ± "); printf "%s%s", arr[1], (i<6?" ":"")}}')
          to_decimal(){ awk -v x="$1" 'BEGIN{printf "%.6f", x/100.0}'; }
          VALIDITY=$(to_decimal "$M1")
          UNIQUENESS=$(to_decimal "$M2")
          QUALITY=$(to_decimal "$M3")
          QUALITY2=$(to_decimal "$M4")
          DIVERSITY="$M5"
          SAMPLE_TIME="$M6"
        else
          # 回退到解析第一条 Run 行
          VALIDITY=$(parse_or_na '(?i)Validity=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
          UNIQUENESS=$(parse_or_na '(?i)Uniqueness=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
          DIVERSITY=$(parse_or_na '(?i)Diversity=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
          QUALITY=$(parse_or_na '(?i)Quality=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
          QUALITY2=$(parse_or_na '(?i)Quality2=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
          SAMPLE_TIME=$(parse_or_na '(?i)Time=\K[0-9]+(?:\.[0-9]+)?(?=s)' "$TMP_OUT")
        fi
      else
        # -r == 1：直接解析单次运行的指标
        VALIDITY=$(parse_or_na '(?i)Validity=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
        UNIQUENESS=$(parse_or_na '(?i)Uniqueness=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
        DIVERSITY=$(parse_or_na '(?i)Diversity=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
        QUALITY=$(parse_or_na '(?i)Quality=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
        QUALITY2=$(parse_or_na '(?i)Quality2=\K[0-9]+(?:\.[0-9]+)?' "$TMP_OUT")
        SAMPLE_TIME=$(parse_or_na '(?i)Time=\K[0-9]+(?:\.[0-9]+)?(?=s)' "$TMP_OUT")
      fi

      # CSV 转义（文件名可能包含逗号等字符）
      esc_csv() { sed 's/"/""/g' <<<"$1"; }
      CKPT_ESC="$(esc_csv "$CKPT_FILENAME")"

      # 创建临时CSV文件，仅包含数据行
      TMP_CSV=$(mktemp --suffix=.csv)
      echo "临时CSV文件: $TMP_CSV"
      echo "\"$CKPT_ESC\",$VALIDITY,$UNIQUENESS,$DIVERSITY,$QUALITY,$QUALITY2,$SAMPLE_TIME,$TOTAL_GENERATIONS,$P_VALUE,$L_VALUE,$B_VALUE,$T_VALUE,$R_VALUE,$M_VALUE,$E_VALUE,$GPU_COUNT,$SEED_VALUE" > "$TMP_CSV"

      # 将临时CSV路径添加到共享列表
      (
        flock 201
        echo "$TMP_CSV" >> "$TMP_LIST"
      ) 201>"$TMP_LIST.lock"

      rm -f "$TMP_OUT"

      # 若采样脚本整体失败，仍已写入 NA 指标以保留记录
      if [[ $STATUS -ne 0 ]]; then
        echo "警告：sample.py 对 '$CKPT_FILENAME' 执行失败（已记录 NA）。"
      else
        echo "完成：$CKPT_FILENAME -> 临时保存"
      fi
    done
  ) &
  PIDS+=($!)
done

# 等待所有后台进程完成
for PID in "${PIDS[@]}"; do
  wait "$PID"
done

# 合并临时CSV到最终CSV
if [[ -f "$TMP_LIST" ]]; then
  while IFS= read -r tmp_csv; do
    if [[ -f "$tmp_csv" ]]; then
      cat "$tmp_csv" >> "$OUTPUT_CSV"
      rm -f "$tmp_csv"
    fi
  done < "$TMP_LIST"
  rm -f "$TMP_LIST" "$TMP_LIST.lock"
fi

echo "全部完成：共处理 ${#CKPTS[@]} 个检查点。输出 CSV：$OUTPUT_CSV"
