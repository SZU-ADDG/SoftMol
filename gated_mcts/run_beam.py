from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from rdkit import rdBase

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util as _ilu

_utils_path = _ROOT / "utils.py"
if "utils" not in sys.modules and _utils_path.exists():
    _spec = _ilu.spec_from_file_location("utils", str(_utils_path))
    _mod = _ilu.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)
    sys.modules["utils"] = _mod

from gated_mcts.mcts import BD3Sampler, MCTSConfig, MolecularProblemState
from gated_mcts.search_baselines import (
    BeamSearcher,
    DockingBudgetWrapper,
    make_trace_path,
    write_trace_csv,
)
from tokenizer import SmilesTokenizer

rdBase.DisableLog("rdApp.warning")


def _resolve_checkpoint(ckpt: str) -> str:
    actual_ckpt = ckpt
    if not os.path.exists(actual_ckpt):
        try:
            from huggingface_hub import hf_hub_download

            print(f"[INFO] Local checkpoint '{actual_ckpt}' not found. Downloading from Hugging Face Hub...")
            actual_ckpt = hf_hub_download(repo_id="SZU-ADDG/SoftMol", filename=actual_ckpt)
        except ImportError:
            print("[WARN] huggingface_hub is not installed, unable to auto-download.")
        except Exception as e:
            print(f"[WARN] Auto-download failed: {e}")
    return actual_ckpt


def _build_tokenizer(vocab: str) -> SmilesTokenizer:
    tokenizer = SmilesTokenizer(vocab)
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.cls_token = "[CLS]"
    return tokenizer


def _build_model(args) -> BD3Sampler:
    actual_ckpt = _resolve_checkpoint(args.ckpt)
    return BD3Sampler(
        gpu=str(args.device),
        ckpt=str(actual_ckpt),
        vocab=str(args.vocab),
        length=int(args.length),
        block_size=int(args.block_size),
        steps=int(args.steps),
        nucleus=float(args.nucleus),
        temperature=float(args.temperature),
        eval_bsz=int(args.gen_batch_size),
        model_name=str(args.model),
        out=str(args.output_file_path),
        seed=int(args.seed),
    )


def _run_single(model, tokenizer, device, args, seed: int) -> tuple[float, float | None, str | None, str | None]:
    predictor = DockingBudgetWrapper(args.protein, max_oracle_calls=int(args.search_time))
    x = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64).unsqueeze(0).to(device)
    max_steps_blocks = int(args.length) // max(1, int(args.block_size))

    initial_state = MolecularProblemState(
        model=model,
        tokenizer=tokenizer,
        predictor=predictor,
        enable_qed_sa_gate=not bool(args.disable_qed_sa_gate),
        use_geam_score=bool(args.use_geam_score),
        cur_molecule=x,
        max_steps=int(max_steps_blocks),
    )
    config = MCTSConfig(
        value_weight=float(args.value_weight),
        search_time=int(args.search_time),
        min_terminals=int(args.min_terminals),
        max_split_depth=int(args.max_split_depth),
        init_children=int(args.init_children),
        n_total_children=int(args.n_total_children),
        c_param=float(args.c_param),
        width_increase_factor=int(args.width_increase_factor),
        add_value_weight=float(args.add_value_weight),
        n_simulations=int(args.n_simulations),
        fastrollout_weight=float(args.fastrollout_weight),
        greedy_path=bool(args.greedy_path),
        max_n_repeat=int(args.max_n_repeat),
        diversity_threshold=float(args.diversity_threshold),
        max_resample_on_empty=int(args.max_resample_on_empty),
    )

    searcher = BeamSearcher(initial_state, config)
    t0 = time.time()
    with torch.no_grad():
        rv, smi, sentence = searcher.run()
    elapsed = time.time() - t0
    return elapsed, rv, smi, sentence, searcher


def _write_results(args, elapsed: float, rv, smi, sentence, seed: int) -> None:
    out_csv = os.path.join(args.output_file_path, args.output_file_name)
    mode = "a" if os.path.exists(out_csv) else "w"
    with open(out_csv, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("rv,smi,cur_sentence,elapsed_time,seed\n")
        f.write(f'{rv},"{smi}","{sentence}",{elapsed},{seed}\n')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_path", default="./mcts_output/", help="Output directory path (auto created)")
    parser.add_argument("--output_file_name", default="beam.csv", help="Output filename (with extension)")
    parser.add_argument("--device", default="0", help="GPU ID, e.g., 0 or 0,1")
    parser.add_argument("--sample_num", default="1", help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation process")
    parser.add_argument("--ckpt", default="weights/89M-epoch6-best.ckpt", help="checkpoint path (.ckpt)")
    parser.add_argument("--vocab", default="vocab_V2.txt", help="Vocabulary path")
    parser.add_argument("--length", type=int, default=512, help="Model sequence length")
    parser.add_argument("--block_size", type=int, default=8, help="Block size")
    parser.add_argument("--steps", type=int, default=128, help="Diffusion steps T")
    parser.add_argument("-p", "--nucleus", type=float, default=1.0, help="Nucleus sampling threshold p")
    parser.add_argument("--temperature", type=float, default=1.1, help="Sampling temperature")
    parser.add_argument("--gen_batch_size", type=int, default=64, help="Expansion batch candidates")
    parser.add_argument("--model", type=str, default="small-89M", help="Model config name (corresponds to configs/model/<name>.yaml)")
    parser.add_argument("--protein", type=str, default="parp1", choices=["fa7", "parp1", "5ht1b", "jak2", "braf", "6GL8", "1UWH", "7OTE", "1KKQ", "5WFD", "7WC7", "8JJL", "7D42", "7S1S", "6AZV"], help="Protein target for docking")

    parser.add_argument("--value_weight", type=float, default=0.0, help="Weight of value in total reward")
    parser.add_argument("--search_time", type=int, default=1000, help="Docking oracle budget (actual docking calls upper bound)")
    parser.add_argument("--min_terminals", type=int, default=-1, help="Compatibility only; unused in beam baseline")
    parser.add_argument("--max_split_depth", type=int, default=100, help="Compatibility only; unused in beam baseline")
    parser.add_argument("--init_children", type=int, default=20, help="Beam width for the first layer")
    parser.add_argument("--n_total_children", type=int, default=8, help="Beam width for later layers")
    parser.add_argument("--c_param", type=float, default=2.1, help="Compatibility only; unused in beam baseline")
    parser.add_argument("--width_increase_factor", type=int, default=2, help="Compatibility only; unused in beam baseline")
    parser.add_argument("--add_value_weight", type=float, default=0.0, help="Compatibility only")
    parser.add_argument("--n_simulations", type=int, default=1, help="Number of rollout simulations used for scoring")
    parser.add_argument("--fastrollout_weight", type=float, default=1.0, help="Fast rollout (Simulation) weight")
    parser.add_argument("--greedy_path", action="store_true", help="Compatibility only; unused in beam baseline")
    parser.add_argument("--max_n_repeat", type=int, default=5, help="Compatibility only")
    parser.add_argument("--diversity_threshold", type=float, default=0.6, help="Compatibility only")
    parser.add_argument("--max_resample_on_empty", type=int, default=5, help="Max resampling rounds when no new candidates are found")
    parser.add_argument("--disable_qed_sa_gate", action="store_true", help="QED/SA gate is enabled by default; pass this parameter to disable it")
    parser.add_argument("--use_geam_score", action="store_true", help="Use GEAM-style score instead of the default docking-only reward")
    parser.add_argument("--trace_path", type=str, default=None, help="Search trace save path (directory or filename base, no index; unified to CSV)")
    return parser.parse_args()


def main():
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda:0")
    os.makedirs(args.output_file_path, exist_ok=True)

    tokenizer = _build_tokenizer(args.vocab)
    model = _build_model(args)

    sample_num = int(args.sample_num)
    for idx in range(sample_num):
        print("sample:", idx + 1)
        elapsed, rv, smi, sentence, searcher = _run_single(model, tokenizer, device, args, seed=int(args.seed))
        _write_results(args, elapsed, rv, smi, sentence, seed=int(args.seed))

        try:
            if args.trace_path is not None and len(str(args.trace_path)) > 0:
                trace_target = Path(args.trace_path)
                if trace_target.suffix.lower() == ".csv":
                    base_dir = args.output_file_path
                    base_name = trace_target.name
                else:
                    base_dir = str(trace_target)
                    base_name = "trace"
                trace_file = make_trace_path(base_dir, base_name, idx + 1)
            else:
                trace_file = make_trace_path(args.output_file_path, "trace", idx + 1)
            write_trace_csv(trace_file, searcher)
        except Exception as e:
            print(f"[warn] Failed to save trace: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
