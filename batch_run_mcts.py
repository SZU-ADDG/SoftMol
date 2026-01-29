"""
Simplified batch startup script: fix a seed, loop runs times, asynchronously launch gated_mcts/run_mcts.py,
and save results to <base_output_dir>/seed{seed}/mcts_job_{i}_seed_{seed}.csv.

Usage example:
python batch_run_mcts.py \
  --base_output_dir ./mcts_output/ \
  --runs 9 --device 0 --seed 42 \
  --ckpt weights/89M-epoch6-best.ckpt \
  --vocab vocab_V2.txt --length 512 --block_size 8  --gen_batch_size 64 --model small-89M  \
  --sample_num 42 --protein parp1 --seed 42 \
  --search_time 1000 --c_param 2.1 --init_children 20 --n_total_children 8 --max_split_depth 100  -p 1  --temperature 1.1
"""

import os
import sys
import time
import shlex
import uuid
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Simplified: fix seed, run gated_mcts/run_mcts.py runs times asynchronously')
    parser.add_argument('--base_output_dir', type=str, default='./mcts_output', help='Root output directory, e.g. ./mcts_output')
    parser.add_argument('--runs', type=int, default=9, help='Number of launches (loop count)')
    parser.add_argument('--delay', type=float, default=2.0, help='Interval in seconds between adjacent launches')
    parser.add_argument('--seed', type=int, required=True, help='Fixed seed (used for script and run_mcts.py)')
    parser.add_argument('--device', type=str, required=True, help='GPU device ID string, e.g. 0 or 2')
    parser.add_argument('--ckpt',  default='weights/89M-epoch6-best.ckpt', help='checkpoint path (.ckpt)')
    parser.add_argument('--vocab', type=str, default='vocab_V2.txt', help='Vocabulary path')
    parser.add_argument('--length', type=int, default=512, help='Sequence length')
    parser.add_argument('--block_size', type=int, default=8, help='Block size')
    parser.add_argument('--steps', type=int, default=128, help='Diffusion steps T')
    parser.add_argument('-p', '--nucleus', type=float, default=1.0, help='Nucleus sampling threshold p')
    parser.add_argument('--temperature', type=float, default=1.1, help='Sampling temperature')
    parser.add_argument('--gen_batch_size', type=int, default=64, help='Expansion batch candidates count')
    parser.add_argument('--model', type=str, default='small-89M', help='Model config name')
    parser.add_argument('--sample_num', type=int, default=1, help='Number of samples per job')
    parser.add_argument('--protein', type=str, default='parp1', choices=['braf', 'jak2', '5ht1b', 'parp1', 'fa7'], help='Protein target for docking')

    # MCTSConfig related
    parser.add_argument('--value_weight', type=float, default=0.0)
    parser.add_argument('--search_time', type=int, default=1000)
    parser.add_argument('--min_terminals', type=int, default=-1)
    parser.add_argument('--max_split_depth', type=int, default=100)
    parser.add_argument('--init_children', type=int, default=20)
    parser.add_argument('--n_total_children', type=int, default=8)
    parser.add_argument('--c_param', type=float, default=2.1)
    parser.add_argument('--width_increase_factor', type=int, default=2)
    parser.add_argument('--add_value_weight', type=float, default=0.0)
    parser.add_argument('--n_simulations', type=int, default=1)
    parser.add_argument('--fastrollout_weight', type=float, default=1.0)
    parser.add_argument('--greedy_path', action='store_true')
    parser.add_argument('--max_n_repeat', type=int, default=5)
    parser.add_argument('--diversity_threshold', type=float, default=0.6)
    parser.add_argument('--max_resample_on_empty', type=int, default=5)
    parser.add_argument('--disable_qed_sa_gate', action='store_true', help='QED/SA gate is enabled by default; pass this parameter to disable it (integration occurs only when QED > 0.5 and SA_raw < 5.0).')
    parser.add_argument('--trace_path', type=str, default=None, help='Search trace save path (directory or filename base, no index; consistently written as CSV)')
    return parser.parse_args()


def main():
    args = parse_args()

    start_time = time.time()

    base_output_dir = Path(args.base_output_dir) / args.protein
    output_dir = base_output_dir / f'seed{args.seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Output directory: {output_dir}')
    print(f'[INFO] Fixed seed: {args.seed}, runs: {args.runs}, delay: {args.delay}s, device: {args.device}')

    python_exec = sys.executable or 'python'
    base_cmd = [
        python_exec, 'gated_mcts/run_mcts.py',
        '--ckpt', args.ckpt,
        '--vocab', args.vocab,
        '--length', str(args.length),
        '--block_size', str(args.block_size),
        '--steps', str(args.steps),
        '--nucleus', str(args.nucleus),
        '--temperature', str(args.temperature),
        '--gen_batch_size', str(args.gen_batch_size),
        '--model', args.model,
        '--sample_num', str(args.sample_num),
        '--protein', args.protein,
        '--output_file_path', str(output_dir),
        '--value_weight', str(args.value_weight),
        '--search_time', str(args.search_time),
        '--min_terminals', str(args.min_terminals),
        '--max_split_depth', str(args.max_split_depth),
        '--init_children', str(args.init_children),
        '--n_total_children', str(args.n_total_children),
        '--c_param', str(args.c_param),
        '--width_increase_factor', str(args.width_increase_factor),
        '--add_value_weight', str(args.add_value_weight),
        '--n_simulations', str(args.n_simulations),
        '--fastrollout_weight', str(args.fastrollout_weight),
        '--max_n_repeat', str(args.max_n_repeat),
        '--diversity_threshold', str(args.diversity_threshold),
        '--max_resample_on_empty', str(args.max_resample_on_empty),
    ]
    if args.disable_qed_sa_gate:
        base_cmd.append('--disable_qed_sa_gate')

    if args.trace_path is not None and len(str(args.trace_path)) > 0:
        base_cmd += [
            '--trace_path', str(args.trace_path),
        ]

    procs = []
    for i in range(args.runs):
        rand_tag = uuid.uuid4().hex[:8]
        out_name = f'mcts_job_{i}_seed_{args.seed}_sample_num_{args.sample_num}_{rand_tag}.csv'
        cmd = base_cmd + [
            '--device', str(args.device),
            '--seed', str(args.seed),
            '--output_file_name', out_name,
        ]
        if args.greedy_path:
            cmd.append('--greedy_path')

        cmd_str = ' '.join(shlex.quote(x) for x in cmd)
        print(f'[LAUNCH {i+1}/{args.runs}] {cmd_str}')

        p = subprocess.Popen(cmd_str, shell=True)
        procs.append(p)

        if i != args.runs - 1:
            time.sleep(args.delay)

    elapsed = time.time() - start_time
    print(f'Launched {args.runs} tasks, elapsed {elapsed:.2f}s (not waiting for subprocesses to complete).')


if __name__ == '__main__':
    main()
