#!/usr/bin/env bash
# 简单的流程：先合并再评估。

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_eval_mcts.sh [-d dir] [-p prefix] [-o output] [-t target] [-s tools_dir]
  -d 目录（默认 .）
  -p CSV 文件名前缀（默认 mcts_job）
  -o 合并后 CSV 输出路径（默认 <dir>/eval_merged.csv）
  -t 靶标蛋白名（parp1/fa7/5ht1b/braf/jak2，默认 braf）
  -s tools 目录（可选，需包含 merge_mcts.py / eval_mcts.py）
EOF
}

DIR="."
PREFIX="mcts_job"
OUTPUT=""
TARGET="braf"
TOOLS_DIR=""

while getopts "d:p:o:t:s:h" opt; do
  case "$opt" in
    d) DIR="$OPTARG" ;;
    p) PREFIX="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    t) TARGET="$OPTARG" ;;
    s) TOOLS_DIR="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

if [ $# -gt 0 ]; then
  usage
  exit 1
fi

if [ -z "$OUTPUT" ]; then
  OUTPUT="$DIR/eval_merged.csv"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

if [ -z "$TOOLS_DIR" ]; then
  CANDIDATES=(
    "$REPO_ROOT/tools"
    "$WORKSPACE_ROOT/bd3lms-1.0/tools"
    "$WORKSPACE_ROOT/bd3lms-1.0-backup/tools"
  )
  for c in "${CANDIDATES[@]}"; do
    if [ -f "$c/merge_mcts.py" ] && [ -f "$c/eval_mcts.py" ]; then
      TOOLS_DIR="$c"
      break
    fi
  done
fi

if [ -z "$TOOLS_DIR" ] || [ ! -f "$TOOLS_DIR/merge_mcts.py" ] || [ ! -f "$TOOLS_DIR/eval_mcts.py" ]; then
  echo "错误：找不到 merge_mcts.py / eval_mcts.py。" >&2
  echo "请用 -s 指定 tools 目录，例如：" >&2
  echo "  -s $WORKSPACE_ROOT/bd3lms-1.0/tools" >&2
  exit 1
fi

python "$TOOLS_DIR/merge_mcts.py" -d "$DIR" -o "$OUTPUT" --prefix "$PREFIX"
python "$TOOLS_DIR/eval_mcts.py" -i "$OUTPUT" -t "$TARGET"
