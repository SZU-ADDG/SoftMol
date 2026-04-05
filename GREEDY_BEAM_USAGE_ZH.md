# SoftMol Greedy / Beam 消融使用说明

本文档介绍如何在当前 SoftMol 仓库中使用新增的 `Greedy Search` 和 `Beam Search` 两种搜索基线，用于替换原有 `MCTS` 做消融实验。

## 1. 新增脚本

本次新增了 4 个脚本：

- `gated_mcts/run_greedy.py`
- `gated_mcts/run_beam.py`
- `batch_run_greedy.py`
- `batch_run_beam.py`

它们的参数风格尽量与现有的：

- `gated_mcts/run_mcts.py`
- `batch_run_mcts.py`

保持一致，因此原有命令通常只需要把脚本名替换掉即可。

## 2. 与 MCTS 的主要区别

### 2.1 Greedy Search

- 每一步只保留 1 条当前最优路径
- 不做树回溯
- 不保留多个分支
- 为了满足固定 docking 预算，允许在同一步上重复采样多个候选，再从中选最优 child 前进

### 2.2 Beam Search

- 第 1 层保留 `init_children` 条候选
- 第 2 层及以后保留 `n_total_children` 条候选
- 每层扩展后按分数排序，只保留 top-K

### 2.3 `search_time` 的语义

这是和 `MCTS` 最大的不同。

- 在 `MCTS` 中，`search_time` 表示搜索迭代次数上限
- 在 `Greedy` / `Beam` 中，`search_time` 表示真实的 docking oracle 预算

也就是说：

```bash
--search_time 1000
```

在 `Greedy` / `Beam` 中的含义是：

- 最多并尽量执行 `1000` 次真实 docking

不是：

- 1000 个搜索步
- 1000 个树节点
- 1000 次 block 扩展

## 3. 打分对象是否和 MCTS 一致

一致。

在默认配置下，`Greedy` / `Beam` 和 `MCTS` 都是：

- 先从当前 partial prefix 生成一个完整分子
- 再把这个完整分子送去 docking
- 再用 docking reward 给当前候选打分

因此对于你的消融实验来说，这两种基线和 `MCTS` 在“真实 docking 的对象”上是对齐的。

## 4. feasibility gate / QED-SA gate

当前仓库里的 feasibility gate 对应已有的 `QED/SA gate`。

默认行为：

- 不传 `--disable_qed_sa_gate` 时，默认开启

只有显式加上下面这个参数才会关闭：

```bash
--disable_qed_sa_gate
```

## 5. 运行前准备

进入项目目录：

```bash
cd /share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol
```

使用 `softmol` 环境：

```bash
conda run -n softmol python -V
```

确保以下文件存在：

- `weights/best.ckpt`
- `vocab_V2.txt`
- `gated_mcts/utils/docking/qvina02`
- 对应蛋白的 `pdbqt` 文件，例如 `gated_mcts/utils/docking/6GL8.pdbqt`

## 6. 单次运行

### 6.1 Greedy

```bash
cd /share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol

conda run -n softmol python gated_mcts/run_greedy.py \
  --output_file_path ./mcts_output/ablation/greedy \
  --output_file_name greedy.csv \
  --device 0 \
  --sample_num 1 \
  --seed 42 \
  --ckpt weights/best.ckpt \
  --vocab vocab_V2.txt \
  --length 100 \
  --block_size 8 \
  --steps 128 \
  --gen_batch_size 10 \
  --model small-89M \
  --protein 6GL8 \
  --search_time 1000 \
  --init_children 20 \
  --n_total_children 8 \
  --temperature 1.1 \
  -p 1 \
  --trace_path ./mcts_output/ablation/greedy_traces
```

### 6.2 Beam

```bash
cd /share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol

conda run -n softmol python gated_mcts/run_beam.py \
  --output_file_path ./mcts_output/ablation/beam \
  --output_file_name beam.csv \
  --device 0 \
  --sample_num 1 \
  --seed 42 \
  --ckpt weights/best.ckpt \
  --vocab vocab_V2.txt \
  --length 100 \
  --block_size 8 \
  --steps 128 \
  --gen_batch_size 10 \
  --model small-89M \
  --protein 6GL8 \
  --search_time 1000 \
  --init_children 20 \
  --n_total_children 8 \
  --temperature 1.1 \
  -p 1 \
  --trace_path ./mcts_output/ablation/beam_traces
```

## 7. 批量运行

### 7.1 Greedy 批量运行

如果你原来跑 MCTS 的命令是：

```bash
conda run -n softmol python batch_run_mcts.py ...
```

那么 Greedy 版本通常只需要替换成：

```bash
conda run -n softmol python batch_run_greedy.py ...
```

示例：

```bash
cd /share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol

conda run -n softmol python batch_run_greedy.py \
  --base_output_dir ./mcts_output/beyond_affinity/SMILES-pt/ \
  --runs 9 \
  --device 0 \
  --seed 42 \
  --ckpt weights/best.ckpt \
  --vocab vocab_V2.txt \
  --length 100 \
  --block_size 8 \
  --gen_batch_size 10 \
  --model small-89M \
  --sample_num 3 \
  --protein 6GL8 \
  --search_time 1000 \
  --c_param 2.1 \
  --init_children 20 \
  --n_total_children 8 \
  --max_split_depth 100 \
  -p 1 \
  --temperature 1.1
```

### 7.2 Beam 批量运行

```bash
cd /share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol

conda run -n softmol python batch_run_beam.py \
  --base_output_dir ./mcts_output/beyond_affinity/SMILES-pt/ \
  --runs 9 \
  --device 0 \
  --seed 42 \
  --ckpt weights/best.ckpt \
  --vocab vocab_V2.txt \
  --length 100 \
  --block_size 8 \
  --gen_batch_size 10 \
  --model small-89M \
  --sample_num 3 \
  --protein 6GL8 \
  --search_time 1000 \
  --c_param 2.1 \
  --init_children 20 \
  --n_total_children 8 \
  --max_split_depth 100 \
  -p 1 \
  --temperature 1.1
```

## 8. 建议直接保留不变的参数

如果你的目标是“只替换搜索方法，其他条件尽量不变”，建议保留下面这些参数与 MCTS 完全一致：

- `--ckpt`
- `--vocab`
- `--length`
- `--block_size`
- `--steps`
- `--gen_batch_size`
- `--model`
- `--sample_num`
- `--protein`
- `--temperature`
- `-p`
- `--init_children`
- `--n_total_children`
- `--search_time`

其中要特别注意：

- `--block_size 8` 可以原样保留
- `--search_time 1000` 在 `Greedy` / `Beam` 中表示 1000 次 docking 预算

## 9. 参数说明

### 9.1 核心参数

- `--search_time`
  - 在 `Greedy` / `Beam` 中表示真实 docking 次数预算
- `--init_children`
  - Greedy 中表示 root step 的基础候选宽度
  - Beam 中表示第 1 层保留宽度
- `--n_total_children`
  - Greedy 中表示后续每步基础候选宽度
  - Beam 中表示第 2 层及以后保留宽度
- `--n_simulations`
  - 每个候选用于 rollout 打分的完整分子生成次数
- `--fastrollout_weight`
  - rollout 分数的权重

### 9.2 兼容保留但在基线中不主导逻辑的参数

下面这些参数保留是为了和原脚本接口兼容：

- `--c_param`
- `--width_increase_factor`
- `--greedy_path`
- `--max_split_depth`
- `--min_terminals`
- `--add_value_weight`
- `--max_n_repeat`
- `--diversity_threshold`

它们不会像在 `MCTS` 中那样直接决定树搜索行为。

## 10. 输出文件说明

### 10.1 结果 CSV

结果文件字段与原 `run_mcts.py` 保持一致：

- `rv`
- `smi`
- `cur_sentence`
- `elapsed_time`
- `seed`

### 10.2 Trace CSV

如果传入：

```bash
--trace_path ./some_dir
```

则会在该目录下输出：

- `trace_1.csv`
- `trace_2.csv`
- ...

如果传入的是 `.csv` 文件名，则会把它当作文件名前缀。

Trace 中会额外记录：

- `oracle_calls`

用于确认 docking 预算是否真的被消耗到了指定值。

## 11. 如何确认预算生效

例如：

```bash
--search_time 1000
```

运行结束后，你可以查看 trace 最后一行附近的 `oracle_calls`。

如果搜索空间没有提前耗尽，那么：

- `oracle_calls` 应接近或等于 `1000`

如果没有达到 1000，常见原因是：

- 当前搜索空间无法继续产生可评估候选
- 候选长期无法通过 gate
- 生成结果大量无效，无法进入真实 docking

这时 trace 中通常会出现：

- `stop_exhausted`
- `stop_no_oracle_progress`
- `stop_budget`

## 12. 常见问题

### 12.1 为什么日志里会有很多 RDKit 的 SMILES Parse Error？

这是生成过程中出现了无效或不完整 SMILES，属于搜索过程中的正常现象。只要脚本没有中断，通常不影响整体运行。

### 12.2 为什么偶尔 qvina 会报错？

少数候选可能在 3D 构象生成或 docking 阶段失败。当前 docking 代码会把这类样本按失败分数处理，脚本一般不会因此崩溃。

### 12.3 默认 gate 开了吗？

开了。

只有显式传：

```bash
--disable_qed_sa_gate
```

才会关闭。

## 13. 推荐用法

如果你的目标是和现有 MCTS 做公平消融，推荐直接采用下面的替换策略：

- `MCTS` 命令改成 `batch_run_greedy.py`
- 或改成 `batch_run_beam.py`
- 其余超参数尽量不变

也就是：

- 保持同一个模型
- 保持同一个蛋白
- 保持同一个 `block_size`
- 保持同一个生成温度和采样设置
- 保持同一个 `search_time`

这样最容易把性能差异解释为“搜索策略不同”，而不是“其他配置变化”。
