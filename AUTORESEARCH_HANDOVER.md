# AutoResearch 交接文档（PMO / troglitazone_rediscovery）

更新时间：2026-03-28  
项目根目录：`/share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol`

## 1. 已完成的开发工作

### 新增脚本

- `tool/autotune_pmo_troglitazone.py`
  - 异步多卡自动调参器，调用 `gated_mcts/run_pmo_mcts.py`。
  - 在代码中强制固定：
    - `--max_oracle_calls 10000`
    - `--freq_log 100`
  - 优化目标：
    - 主目标：`auc_top10`
    - 次级排序：`top10`，再 `top1`
  - 可调参数：
    - `block_size`
    - `nucleus`
    - `temperature`
    - `init_children`
    - `n_total_children`
    - `c_param`
    - `width_increase_factor`
  - 支持通过 `*_values` 传入 CSV 形式的“收缩搜索空间”。

- `tool/start_troglitazone_autotune.sh`
  - 后台启动脚本。
  - 使用 `setsid + conda run --no-capture-output`，避免会话退出导致任务中断。
  - 输出：
    - `tuner.pid`
    - `tuner_stdout.log`

- `tool/watch_autotune_and_chain.py`
  - 监控某一轮调参任务。
  - 当前轮结束后，自动分析 top trial，收缩搜索空间并拉起下一轮。
  - 当前使用 `--chain_rounds 3` 串联多轮。

## 2. 实验状态（已完成 / 进行中）

### 基线实验（已完成）

- 目录：`results/pmo/main01/bs2_10000`
- 文件：`pmo_metrics_seed42.csv`
- `troglitazone_rediscovery` 基线：
  - `auc_top10 = 0.2546098384`
  - `top10 = 0.2644148934`

### 自动调参 Round1（已完成）

- 目录：`results/pmo/autotune_troglitazone_20260328_010644`
- 状态：`96/96` 完成，`96` 成功
- 最优 trial：`trial_0013`
  - `auc_top10 = 0.3052947134`
  - `top10 = 0.4091499090`
  - `top1 = 0.4636363636`
  - 参数：
    - `block_size=4`
    - `nucleus=1.0`
    - `temperature=1.2`
    - `init_children=16`
    - `n_total_children=10`
    - `c_param=3.2`
    - `width_increase_factor=2`

### 自动调参 Round2（进行中）

- 目录：`results/pmo/autotune_troglitazone_20260328_010644_focus_r2_20260328_085830`
- 状态（最近一次检查）：
  - `launched=64`
  - `completed=48`
  - `running=16`
  - `max_trials=72`
- 当前最好 trial：`trial_0041`
  - `auc_top10 = 0.3237158051`
  - `top10 = 0.3902697188`
  - `top1 = 0.4395604396`
  - 参数：
    - `block_size=4`
    - `nucleus=1.0`
    - `temperature=1.3`
    - `init_children=12`
    - `n_total_children=10`
    - `c_param=3.2`
    - `width_increase_factor=2`

## 3. 参数影响结论（经验总结）

基于 Round1 完整结果 + Round2 已完成结果：

- `block_size`：`4` 明显强于 `2`。
- `nucleus`：`1.0` 效果最好，`0.95` 次之。
- `temperature`：`1.2 / 1.3` 更容易出最优。
- `init_children`：`12~16` 更稳，较大值通常不占优。
- `n_total_children`：`10` 最稳定。
- `c_param`：`3.2` 在高分配置中反复出现。
- `width_increase_factor`：`2` 最稳。

## 4. 当前运行进程与监控

### 进程链（最近观测）

- 监控器：
  - `tool/watch_autotune_and_chain.py --chain_rounds 3`
  - 观测 PID：`19993`（wrapper）/ `20006`（child）
- Round2 调参器：
  - `tool/autotune_pmo_troglitazone.py --output_root results/pmo/autotune_troglitazone_20260328_010644_focus_r2_20260328_085830 ...`
  - 观测 PID：`96834`（wrapper）/ `96845`（child）

注意：PID 可能变化，交接时以 `ps` 实时结果为准。

### 常用命令

- 看监控日志：
  - `tail -f results/pmo/autotune_troglitazone_20260328_010644/watcher_stdout.log`
- 看 Round2 日志：
  - `tail -f results/pmo/autotune_troglitazone_20260328_010644_focus_r2_20260328_085830/tuner_stdout.log`
- 看 Round2 状态：
  - `cat results/pmo/autotune_troglitazone_20260328_010644_focus_r2_20260328_085830/state.json`
- 看 Round2 排行榜：
  - `sed -n '1,40p' results/pmo/autotune_troglitazone_20260328_010644_focus_r2_20260328_085830/leaderboard.csv`

## 5. 接手同学操作清单

1. 检查进程是否都在：
   - `ps -ef | rg "watch_autotune_and_chain.py|autotune_pmo_troglitazone.py"`
2. 确保没有重复 watcher（同一根目录只保留一个）。
3. 等 Round2 结束（`state.running_trials == 0`）。
4. 确认 watcher 自动拉起 Round3：
   - 目录应类似：`results/pmo/autotune_troglitazone_20260328_010644_focus_r3_<timestamp>`
5. 每轮结束后输出简报：
   - best config
   - best `auc_top10/top10/top1`
   - 对基线提升

## 6. 风险与注意事项

- 历史上出现过旧 watcher 残留，已清理；交接后仍需再次检查重复 watcher。
- `state.json` 是快照，不是最终真值；请优先看：
  - `all_trials.jsonl`
  - `leaderboard.csv`
- `max_oracle_calls=10000` 时单 trial 耗时较长，长时间无新完成不一定异常。

## 7. 手动恢复命令（兜底）

### 仅重启 watcher（保持当前轮不变）

```bash
cd /share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol
setsid conda run --no-capture-output -n softmol python -u tool/watch_autotune_and_chain.py \
  --current_output_dir results/pmo/autotune_troglitazone_20260328_010644 \
  --poll_seconds 120 \
  --summary_every_seconds 300 \
  --chain_rounds 3 \
  --next_time_budget_hours 10 \
  --next_max_trials 72 \
  --next_initial_random_trials 12 \
  --next_exploit_probability 0.8 \
  --next_top_pool 6 \
  > results/pmo/autotune_troglitazone_20260328_010644/watcher_stdout.log 2>&1 < /dev/null &
```

### 恢复某一轮 tuner（中断后续跑）

```bash
cd /share/home/tm866079609100000/a875465180/yqw_bd3lms/SoftMol
bash tool/start_troglitazone_autotune.sh \
  --output-dir results/pmo/<目标轮目录> \
  --resume
```

---

交接建议：先看本文件，再看当前 `watcher_stdout.log` 和当前轮 `state.json`，就能快速接上。
