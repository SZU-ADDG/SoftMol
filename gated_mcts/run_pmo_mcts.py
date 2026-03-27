from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from rdkit import rdBase

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Prioritize binding root utils.py before importing sample/dataloader
import importlib.util as _ilu

_utils_path = _ROOT / "utils.py"
if "utils" not in sys.modules and _utils_path.exists():
    _spec = _ilu.spec_from_file_location("utils", str(_utils_path))
    _mod = _ilu.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)
    sys.modules["utils"] = _mod

from tokenizer import SmilesTokenizer
from gated_mcts.mcts import BD3Sampler, MCTS, MCTSConfig, MolecularProblemState
from gated_mcts.utils.pmo_metrics import compute_pmo_metrics
from gated_mcts.utils.pmo_oracle_adapter import PMO_ORACLES, PMOOracleAdapter

rdBase.DisableLog("rdApp.warning")


def _write_trace_csv(trace_path: str, mcts_obj) -> None:
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    events = mcts_obj.get_trace() if hasattr(mcts_obj, "get_trace") else []
    fieldnames = [
        "iter",
        "type",
        "node_id",
        "depth",
        "path",
        "rv",
        "reward",
        "n_visits",
        "q_sum",
        "n_children",
        "terminal",
        "sentence",
        "total_steps",
        "total_rollouts",
        "total_requests",
        "oracle_calls",
    ]
    with open(trace_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            row = {k: ev.get(k, "") for k in fieldnames}
            if isinstance(row.get("path"), list):
                row["path"] = "[" + ",".join(str(x) for x in row["path"]) + "]"
            writer.writerow(row)


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


def _save_history_csv(path: Path, predictor: PMOOracleAdapter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["call_idx", "smiles", "score"])
        for rec in predictor.history:
            writer.writerow([int(rec.call_idx), rec.smiles, float(rec.score)])


def _save_topk_csv(path: Path, predictor: PMOOracleAdapter, k: int) -> None:
    rows = predictor.topk(k=k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "call_idx", "smiles", "score"])
        for rank, rec in enumerate(rows, start=1):
            writer.writerow([rank, int(rec.call_idx), rec.smiles, float(rec.score)])


def _save_standard_topk_outputs(output_dir: Path, prefix: str, predictor: PMOOracleAdapter, custom_k: int) -> dict:
    top10_path = output_dir / f"{prefix}_top10.csv"
    top100_path = output_dir / f"{prefix}_top100.csv"
    _save_topk_csv(top10_path, predictor, 10)
    _save_topk_csv(top100_path, predictor, 100)

    custom_k = int(custom_k)
    if custom_k == 10:
        topk_path = top10_path
    elif custom_k == 100:
        topk_path = top100_path
    else:
        topk_path = output_dir / f"{prefix}_top{custom_k}.csv"
        _save_topk_csv(topk_path, predictor, custom_k)

    return {
        "top10_path": top10_path,
        "top100_path": top100_path,
        "topk_path": topk_path,
    }


def _run_single(args) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device("cuda:0")

    actual_ckpt = _resolve_checkpoint(args.ckpt)

    tokenizer = SmilesTokenizer(args.vocab)
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.cls_token = "[CLS]"

    model = BD3Sampler(
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
        out=str(args.output_dir),
        seed=int(args.seed),
    )

    predictor = PMOOracleAdapter(
        oracle_name=args.oracle_name,
        max_oracle_calls=int(args.max_oracle_calls),
        freq_log=int(args.freq_log),
    )

    x = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64).unsqueeze(0).to(device)
    max_steps_blocks = int(args.length) // max(1, int(args.block_size))
    initial_state = MolecularProblemState(
        model=model,
        tokenizer=tokenizer,
        predictor=predictor,
        enable_qed_sa_gate=False,
        reward_mode="pmo",
        cur_molecule=x,
        max_steps=int(max_steps_blocks),
    )
    mcts_config = MCTSConfig(
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

    t0 = time.time()
    mcts = MCTS(initial_state, mcts_config)
    with torch.no_grad():
        rv, smi, cur_sentence = mcts.run()
    elapsed_time = time.time() - t0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.oracle_name}_seed{args.seed}"

    history_path = output_dir / f"{prefix}_history.csv"
    metrics_path = output_dir / f"{prefix}_metrics.json"

    _save_history_csv(history_path, predictor)
    topk_outputs = _save_standard_topk_outputs(
        output_dir=output_dir,
        prefix=prefix,
        predictor=predictor,
        custom_k=int(args.save_topk),
    )
    top10_path = topk_outputs["top10_path"]
    top100_path = topk_outputs["top100_path"]
    topk_path = topk_outputs["topk_path"]

    metrics = compute_pmo_metrics(
        predictor.buffer,
        freq_log=int(args.freq_log),
        max_oracle_calls=int(args.max_oracle_calls),
    )
    metrics.update(
        {
            "oracle_name": args.oracle_name,
            "seed": int(args.seed),
            "max_oracle_calls": int(args.max_oracle_calls),
            "freq_log": int(args.freq_log),
            "elapsed_time_sec": float(elapsed_time),
            "best_rv": None if rv is None else float(rv),
            "best_smi": smi,
            "best_sentence": cur_sentence,
            "history_file": history_path.name,
            "top10_file": top10_path.name,
            "top100_file": top100_path.name,
            "topk_file": topk_path.name,
        }
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if args.trace_path:
        trace_path = Path(args.trace_path)
        if trace_path.suffix.lower() != ".csv":
            trace_path = trace_path / f"{prefix}_trace.csv"
        _write_trace_csv(str(trace_path), mcts)

    return {
        "history_path": str(history_path),
        "top10_path": str(top10_path),
        "top100_path": str(top100_path),
        "topk_path": str(topk_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/pmo/softmol_mcts", help="Output directory")
    parser.add_argument("--device", default="0", help="GPU ID, e.g., 0 or 0,1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ckpt", default="weights/89M-epoch6-best.ckpt", help="Checkpoint path (.ckpt)")
    parser.add_argument("--vocab", default="vocab_V2.txt", help="Vocabulary path")
    parser.add_argument("--length", type=int, default=100, help="Model sequence length")
    parser.add_argument("--block_size", type=int, default=2, help="Block size")
    parser.add_argument("--steps", type=int, default=128, help="Diffusion steps T")
    parser.add_argument("-p", "--nucleus", type=float, default=1.0, help="Nucleus sampling threshold p")
    parser.add_argument("--temperature", type=float, default=1.1, help="Sampling temperature")
    parser.add_argument("--gen_batch_size", type=int, default=1, help="Expansion batch candidates")
    parser.add_argument("--model", type=str, default="small-89M", help="Model config name")

    parser.add_argument("--oracle_name", type=str, default="qed", choices=list(PMO_ORACLES), help="PMO oracle task name")
    parser.add_argument("--max_oracle_calls", type=int, default=10000, help="Max unique oracle calls")
    parser.add_argument("--freq_log", type=int, default=100, help="Frequency for AUC logging steps")
    parser.add_argument("--save_topk", type=int, default=100, help="Save top-k molecules")
    parser.add_argument("--trace_path", type=str, default=None, help="Optional trace output path (.csv or directory)")

    # MCTSConfig related
    parser.add_argument("--value_weight", type=float, default=0.0, help="Weight of value in total reward")
    parser.add_argument(
        "--search_time",
        type=int,
        default=100000,
        help="Search iteration upper bound; run stops earlier once oracle budget is reached",
    )
    parser.add_argument("--min_terminals", type=int, default=-1, help="Minimum terminal nodes to find")
    parser.add_argument("--max_split_depth", type=int, default=100, help="Max split depth")
    parser.add_argument("--init_children", type=int, default=20, help="Initial children for root node")
    parser.add_argument("--n_total_children", type=int, default=8, help="Children for non-root nodes")
    parser.add_argument("--c_param", type=float, default=2.1, help="UCB exploration coefficient")
    parser.add_argument("--width_increase_factor", type=int, default=2, help="Adaptive width increase factor")
    parser.add_argument("--add_value_weight", type=float, default=0.0, help="Additional value weight")
    parser.add_argument("--n_simulations", type=int, default=1, help="Number of simulations")
    parser.add_argument("--fastrollout_weight", type=float, default=1.0, help="Fast rollout weight")
    parser.add_argument("--greedy_path", action="store_true", help="Enable greedy path")
    parser.add_argument("--max_n_repeat", type=int, default=5, help="Max repeat limit for same path")
    parser.add_argument("--diversity_threshold", type=float, default=0.6, help="Historical similarity threshold")
    parser.add_argument("--max_resample_on_empty", type=int, default=5, help="Max resample count when empty")

    return parser.parse_args()


def main():
    args = _parse_args()
    result = _run_single(args)
    print(f"[PMO] history: {result['history_path']}")
    print(f"[PMO] top10: {result['top10_path']}")
    print(f"[PMO] top100: {result['top100_path']}")
    print(f"[PMO] topk: {result['topk_path']}")
    print(f"[PMO] metrics: {result['metrics_path']}")
    print(
        "[PMO] n_calls={n_calls} auc_top10={auc_top10:.6f} top10={top10:.6f}".format(
            n_calls=result["metrics"]["n_calls"],
            auc_top10=result["metrics"]["auc_top10"],
            top10=result["metrics"]["top10"],
        )
    )


if __name__ == "__main__":
    main()
