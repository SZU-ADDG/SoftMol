import argparse
import os
import sys
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import omegaconf
import torch

# Avoid directory conflicts
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
try:
    import importlib.util as _ilu
    _utils_path = _ROOT / 'utils.py'
    if _utils_path.exists():
        _spec = _ilu.spec_from_file_location('utils', str(_utils_path))
        _mod = _ilu.module_from_spec(_spec)
        assert _spec is not None and _spec.loader is not None
        _spec.loader.exec_module(_mod)
        sys.modules['utils'] = _mod
except Exception:
    pass

import dataloader
import diffusion
from rdkit import Chem, RDLogger
from tdc import Oracle, Evaluator

RDLogger.DisableLog('rdApp.*')

def _build_config(args, logdir: Path):
    base_cfg = omegaconf.OmegaConf.load("configs/sample.yaml")
    model_yaml = Path("configs/model") / f"{args.model}.yaml"
    model_cfg = omegaconf.OmegaConf.load(str(model_yaml))
    # Total samples determined by -e and -n: total ≈ eval_bsz * num_sample_batches
    sampling_cfg = {
        "logdir": str(logdir),
        "nucleus_p": float(args.nucleus),
        "first_hitting": bool(args.first_hitting),
        "top1": bool(args.top1),
        "prefix": (args.prefix or ""),
        "next_block_only": bool(args.next_block_only),
        "temperature": float(args.temperature),
    }
    num_samples = getattr(args, "num_samples", None)
    if num_samples is not None and num_samples > 0:
        bs = int(args.eval_bsz)
        # Sample at least num_samples, truncate to exact count in Python later
        num_batches = int(np.ceil(num_samples / bs))
        sampling_cfg["num_sample_batches"] = int(num_batches)

    overrides = omegaconf.OmegaConf.create({
        "seed": int(args.seed),
        "block_size": int(args.block_size),
        "algo": {"T": int(args.steps)},
        "model": {"length": int(args.length), "attn_backend": "sdpa"},
        "loader": {"eval_batch_size": int(args.eval_bsz)},
        "sampling": sampling_cfg,
        "eval": {"checkpoint_path": str(Path(args.ckpt).resolve())},
        "data": {"tokenizer_name_or_path": str(Path(args.vocab).resolve())},
    })
    merged = omegaconf.OmegaConf.merge(base_cfg, {"model": model_cfg}, overrides)
    return merged

def _clean_text(samples: list[str]) -> list[str]:
    drop_tokens = ["[BOS]", "[EOS]"]
    cleaned = []
    for s in samples:
        t = s
        for tk in drop_tokens:
            t = t.replace(tk, "")
        cleaned.append(t)
    return cleaned

def format_mean_std(arr: np.ndarray, scale: float, decimals: int) -> str:
    if arr.size == 0:
        return f"{0.0:.{decimals}f} ± {0.0:.{decimals}f}"
    mean = arr.mean() * scale
    std = arr.std(ddof=0) * scale
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def evaluate(smiles_list: list[str], qed_thr: float = 0.6, sa_thr: float = 4.0) -> dict:
    evaluator_div = Evaluator('diversity')
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')

    total = len(smiles_list)
    valid = [s for s in smiles_list if Chem.MolFromSmiles(s)]
    validity = len(valid) / total if total > 0 else 0.0
    uniq = list(set(valid))
    uniqueness = len(uniq) / len(valid) if valid else 0.0
    diversity = float(evaluator_div(uniq)) if len(uniq) > 1 else 0.0

    qed = oracle_qed(valid)
    # SA scoring might error on extreme molecules, handle robustly to avoid breaking evaluation
    try:
        sa = oracle_sa(valid)
    except Exception as e:
        print(f"[WARN] SA oracle error, skipping SA eval: {e}")
        sa = [None] * len(valid)
    ok = sum(1 for q, s in zip(qed, sa) if (q is not None and s is not None and q >= qed_thr and s <= sa_thr))
    ok2 = sum(1 for q, s in zip(qed, sa) if (q is not None and s is not None and q > 0.5 and s < 5.0))
    quality = ok / total if total > 0 else 0.0
    quality2 = ok2 / total if total > 0 else 0.0

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'diversity': diversity,
        'quality': quality,
        'quality2': quality2,
        'total_initial': total,
        'total_valid': len(valid),
        'total_unique': len(uniq),
    }

def generate(*,gpu: str,length: int,block_size: int,ckpt: str,model: str = 'small',steps: int = 128,nucleus: float = 0.95,
            eval_bsz: int = 1,num_samples: Optional[int] = None,seed: int = 42,
            vocab: str = 'vocab.txt',out: str = './sample_logs/simple_sample_',
            prefix: Optional[str] = None,next_block_only: bool = False,
            temperature: float = 1.0,
            first_hitting: bool = True,
            top1: bool = True) -> tuple[list[str], list[str]]:
    gpu_str = str(gpu).strip().lower()
    use_cuda = (gpu_str not in {"cpu", "-1", "none"}) and torch.cuda.is_available()
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.manual_seed(int(seed))

    logdir = Path(out).resolve()
    args_obj = argparse.Namespace(
        model=model, seed=seed, block_size=block_size, steps=steps,
        length=length, eval_bsz=eval_bsz, nucleus=nucleus,
        ckpt=ckpt, vocab=vocab, prefix=prefix, next_block_only=next_block_only,
        temperature=temperature, first_hitting=first_hitting, top1=top1,
        num_samples=num_samples,
    )
    config = _build_config(args_obj, logdir)
    tokenizer = dataloader.get_tokenizer(config)

    mdl = diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False,
        weights_only=False,
    ).to("cuda" if use_cuda else "cpu")

    if config.eval.disable_ema:
        mdl.ema = None

    raw_texts = mdl.restore_model_and_sample(num_steps=config.algo.T)
    # Truncate to exact number if specified
    if num_samples is not None and num_samples > 0:
        raw_texts = raw_texts[:num_samples]
    cleaned_texts = _clean_text(raw_texts)

    return raw_texts, cleaned_texts

def _summarize_results(results: list[dict]):
    if not results:
        return

    print("Validity (%) | Uniqueness (%) | Quality (%) | Quality2 (%) | Diversity | Sampling time (s)")
    print("-" * 78)

    metrics_to_agg = {
        "validity": 100.0,
        "uniqueness": 100.0,
        "quality": 100.0,
        "quality2": 100.0,
        "diversity": 1.0,
        "time": 1.0,
    }
    formatted_stats = []
    for key, scale in metrics_to_agg.items():
        arr = np.array([r.get(key, np.nan) for r in results])
        arr = arr[~np.isnan(arr)]
        decimals = 1 if scale >= 100 else 3
        formatted_stats.append(format_mean_std(arr, scale, decimals))

    print(" | ".join(formatted_stats))

def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone SMILES sampling script")
    parser.add_argument("-g", "--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("-l", "--length", type=int, default=72)  
    parser.add_argument("-b", "--block-size", type=int, default=2)
    parser.add_argument("-c", "--ckpt", type=str, default="weights/89M-epoch6-best.ckpt", help="Path to .ckpt")
    parser.add_argument("-m", "--model", type=str, default="small-89M")
    parser.add_argument("-T", "--steps", type=int, default=300)
    parser.add_argument("-p", "--nucleus", type=float, default=0.95)
    parser.add_argument("-e", "--eval-bsz", type=int, default=1000, help="Evaluation batch size, number of samples per batch")
    parser.add_argument("-n", "--num-samples", type=int, default=None,
                        help="Total samples to generate (default controlled by -e, 1 batch only)")
    parser.add_argument("-r", "--repeat", type=int, default=3, help="Number of runs")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-o", "--out", type=str, default="./sample_logs/sample")
    parser.add_argument("-v", "--vocab", type=str, default="vocab_V2.txt")
    parser.add_argument("--prefix", type=str, default=None, help="Optional: prefix SMILES (excluding [BOS]/[EOS])")
    parser.add_argument("--next-block-only", action="store_true", help="Generate only the next block after prefix and exit")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default 1.0)")
    parser.add_argument("--no-first-hitting", dest="first_hitting", action="store_false",
                        help="Disable first-hitting sampling (enabled by default)")
    parser.add_argument("--no-top1", dest="top1", action="store_false",
                        help="Disable top-1 (Greedy Confidence Decoding) position update (enabled by default)")
    parser.set_defaults(first_hitting=True, top1=True)
    args = parser.parse_args()

    logdir_abs = Path(args.out).resolve()
    logdir_abs.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(args.repeat):
        seed_i = int(args.seed) + i
        t0 = time()
        raw_texts, texts = generate(
            gpu=args.gpu,
            length=args.length,
            block_size=args.block_size,
            ckpt=args.ckpt,
            model=args.model,
            steps=args.steps,
            nucleus=args.nucleus,
            eval_bsz=args.eval_bsz,
            num_samples=args.num_samples,
            seed=seed_i,
            vocab=args.vocab,
            out=args.out,
            prefix=args.prefix,
            next_block_only=args.next_block_only,
            temperature=args.temperature,
            first_hitting=args.first_hitting,
            top1=args.top1,
        )
        elapsed = time() - t0
        txt_path = logdir_abs.with_suffix('.txt')
        txt_path.write_text('\n'.join(texts) + '\n', encoding='utf-8')
        print(f"Generated samples saved to: {txt_path}")
        raw_path = txt_path.with_name(f"{txt_path.stem}_raw{txt_path.suffix}")
        raw_path.write_text('\n'.join(raw_texts) + '\n', encoding='utf-8')
        print(f"Raw samples saved to: {raw_path}")
        met = evaluate(texts)
        met = {**met, 'time': elapsed}
        results.append(met)
        metrics_str = ", ".join(
            f"{k.capitalize()}={v:.3f}" for k, v in met.items() if k != 'time'
        )
        print(f"[Run {i+1}/{args.repeat} | seed={seed_i}] Time={elapsed:.2f}s, {metrics_str}")

    if args.repeat > 1:
        _summarize_results(results)

if __name__ == "__main__":
    main()
