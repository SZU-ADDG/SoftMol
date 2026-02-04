from __future__ import annotations

import torch
import os
import sys
import argparse
import time
import csv
from tqdm import tqdm
from rdkit import rdBase

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Prioritize binding root utils.py before importing sample/dataloader
import importlib.util as _ilu
_utils_path = _ROOT / 'utils.py'
if 'utils' not in sys.modules and _utils_path.exists():
    _spec = _ilu.spec_from_file_location('utils', str(_utils_path))
    _mod = _ilu.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)
    sys.modules['utils'] = _mod

from gated_mcts.utils.docking.docking_utils import DockingVina
from tokenizer import SmilesTokenizer
from gated_mcts.mcts import MCTSConfig, MolecularProblemState, MCTS, BD3Sampler

rdBase.DisableLog('rdApp.warning')


def _write_trace_csv(trace_path: str, mcts_obj):
    """Write MCTS trace to CSV (line by line)."""
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    events = mcts_obj.get_trace() if hasattr(mcts_obj, 'get_trace') else []
    fieldnames = [
        'iter', 'type', 'node_id', 'depth', 'path',
        'rv', 'reward', 'n_visits', 'q_sum', 'n_children', 'terminal',
        'sentence', 'total_steps', 'total_rollouts', 'total_requests'
    ]
    with open(trace_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            row = {k: ev.get(k, '') for k in fieldnames}
            if isinstance(row.get('path'), list):
                row['path'] = '[' + ','.join(str(x) for x in row['path']) + ']'
            writer.writerow(row)

def _make_trace_path(base_dir: str, base_name: str, idx: int) -> str:
    name, ext = os.path.splitext(base_name)
    if ext == '.csv':
        final = f"{name}_{idx}{ext}"
    else:
        final = f"{name}_{idx}.csv"
    return os.path.join(base_dir, final)


def Test(model, tokenizer, device, output_file_path, sample_num, output_file_name: str = 'mcts.csv', seed: int = 42,
         trace_path: str | None = None):
    os.makedirs(output_file_path, exist_ok=True)
    t0 = time.time()
    predictor = DockingVina(opt.protein)
    out_csv = os.path.join(output_file_path, output_file_name)
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('rv,smi,cur_sentence,elapsed_time,seed\n')

    x = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64).unsqueeze(0)
    x = x.to(device)
    sample_num = int(sample_num)
    max_steps_blocks = int(opt.length) // max(1, int(opt.block_size))
    for i in range(sample_num):
        print('sample:', i+1)
        initial_state = MolecularProblemState(
            model=model,
            tokenizer=tokenizer,
            predictor=predictor,
            enable_qed_sa_gate=not bool(opt.disable_qed_sa_gate),
            cur_molecule=x,
            max_steps=int(max_steps_blocks),
        )
        mcts_config = MCTSConfig(
            value_weight=float(opt.value_weight),
            search_time=int(opt.search_time),
            min_terminals=int(opt.min_terminals),
            max_split_depth=int(opt.max_split_depth),
            init_children=int(opt.init_children),
            n_total_children=int(opt.n_total_children),
            c_param=float(opt.c_param),
            width_increase_factor=int(opt.width_increase_factor),
            add_value_weight=float(opt.add_value_weight),
            n_simulations=int(opt.n_simulations),
            fastrollout_weight=float(opt.fastrollout_weight),
            greedy_path=bool(opt.greedy_path),
            max_n_repeat=int(opt.max_n_repeat),
            diversity_threshold=float(opt.diversity_threshold),
            max_resample_on_empty=int(opt.max_resample_on_empty),
        )
        mcts = MCTS(initial_state, mcts_config)
        with torch.no_grad():
            rv, smi, cur_sentence = mcts.run()
            current_elapsed = time.time() - t0
            with open(out_csv, 'a', encoding='utf-8') as f:
                f.write(f'{rv},"{smi}","{cur_sentence}",{current_elapsed},{seed}\n')

            try:
                if trace_path is not None and len(str(trace_path)) > 0:
                    base_dir = output_file_path
                    base_name = 'trace' if os.path.isdir(trace_path) else os.path.basename(trace_path)
                    if os.path.isdir(trace_path):
                        base_dir = trace_path
                    trace_file = _make_trace_path(base_dir, base_name, i+1)
                else:
                    trace_file = _make_trace_path(output_file_path, 'trace', i+1)

                _write_trace_csv(trace_file, mcts)
            except Exception as e:
                print(f"[warn] Failed to save trace: {type(e).__name__}")

    elapsed_time = time.time() - t0
    return elapsed_time


def main_test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    device = torch.device(f'cuda:{0}')

    tokenizer = SmilesTokenizer(opt.vocab)
    tokenizer.bos_token = '[BOS]'
    tokenizer.eos_token = '[EOS]'
    tokenizer.sep_token = '[SEP]'
    tokenizer.pad_token = '[PAD]'
    tokenizer.mask_token = '[MASK]'
    tokenizer.cls_token = '[CLS]'

    model = BD3Sampler(
        gpu=str(opt.device),
        ckpt=str(opt.ckpt),
        vocab=str(opt.vocab),
        length=int(opt.length),
        block_size=int(opt.block_size),
        steps=int(opt.steps),
        nucleus=float(opt.nucleus),
        temperature=float(opt.temperature),
        eval_bsz=int(opt.gen_batch_size),
        model_name=str(opt.model),
        out=str(opt.output_file_path),
        seed=int(opt.seed),
    )
    elapsed_time = Test(
        model,
        tokenizer,
        device,
        output_file_path=args.output_file_path,
        sample_num=args.sample_num,
        output_file_name=args.output_file_name,
        seed=int(opt.seed),
        trace_path=opt.trace_path,
    )
    print(f"running time: {elapsed_time:.4f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file_path', default='./mcts_output/', help='Output directory path (auto created)')
    parser.add_argument('--output_file_name', default='mcts.csv', help='Output filename (with extension)')
    parser.add_argument('--device', default='0', help='GPU ID, e.g., 0 or 0,1')
    parser.add_argument('--sample_num', default='1', help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation process')
    parser.add_argument('--ckpt',  default='weights/89M-epoch6-best.ckpt', help='checkpoint path (.ckpt)')
    parser.add_argument('--vocab', default='vocab_V2.txt', help='Vocabulary path')
    parser.add_argument('--length', type=int, default=512, help='Model sequence length')
    parser.add_argument('--block_size', type=int, default=8, help='Block size')
    parser.add_argument('--steps', type=int, default=128, help='Diffusion steps T')
    parser.add_argument('-p', '--nucleus', type=float, default=1.0, help='Nucleus sampling threshold p')
    parser.add_argument('--temperature', type=float, default=1.1, help='Sampling temperature')
    parser.add_argument('--gen_batch_size', type=int, default=64, help='Expansion batch candidates')
    parser.add_argument('--model', type=str, default='small-89M', help='Model config name (corresponds to configs/model/<name>.yaml)')
    parser.add_argument('--protein', type=str, default='parp1', choices=['braf', 'jak2', '5ht1b', 'parp1', 'fa7'], help='Protein target for docking')

    # MCTSConfig related
    parser.add_argument('--value_weight', type=float, default=0.0, help='Weight of value in total reward')
    parser.add_argument('--search_time', type=int, default=1000, help='Total search rounds (approx upper limit of expansions)')
    parser.add_argument('--min_terminals', type=int, default=-1, help='Minimum terminal nodes to find')
    parser.add_argument('--max_split_depth', type=int, default=100, help='Max split depth (> this depth only single path expansion; -1 for unlimited)')
    parser.add_argument('--init_children', type=int, default=20, help='Initial children for root node; -1 to use n_total_children')
    parser.add_argument('--n_total_children', type=int, default=8, help='Number of children for non-root nodes')
    parser.add_argument('--c_param', type=float, default=2.1, help='UCB exploration coefficient (larger means more exploration)')
    parser.add_argument('--width_increase_factor', type=int, default=2, help='Adaptive width increase factor')
    parser.add_argument('--add_value_weight', type=float, default=0.0, help='Additional value weight (optional)')
    parser.add_argument('--n_simulations', type=int, default=1, help='Number of simulations (after each expansion)')
    parser.add_argument('--fastrollout_weight', type=float, default=1.0, help='Fast rollout (Simulation) weight')
    parser.add_argument('--greedy_path', action='store_true', help='Enable greedy path (root to leaf)')
    parser.add_argument('--max_n_repeat', type=int, default=5, help='Max repeat limit for same path')
    parser.add_argument('--diversity_threshold', type=float, default=0.6, help='[Compatibility] Historical similarity threshold, currently only used to filter "siblings don not repeat"')
    parser.add_argument('--max_resample_on_empty', type=int, default=5, help='Max resampling limit when candidate pool is empty')
    parser.add_argument('--disable_qed_sa_gate', action='store_true', help='QED/SA gate is enabled by default; pass this parameter to disable it (integration occurs only when QED > 0.5 and SA_raw < 5.0).')
    parser.add_argument('--trace_path', type=str, default=None, help='Search trace save path (directory or filename base, no index; unified to CSV)')

    opt = parser.parse_args()

    main_test(opt)
