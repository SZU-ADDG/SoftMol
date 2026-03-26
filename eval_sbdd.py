"""GEAM-aligned evaluation for SoftMol SBDD outputs.

This script follows GEAM-main/eval.py metric definitions as closely as possible:
- Novelty
- #Circle
- Novel hit ratio
- Novel top 5% DS

Only adaptation: input schema is SoftMol CSV with columns ['smi', 'rv'].
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED, RDConfig

try:
    import more_itertools as mit
except Exception:
    mit = None

RDLogger.DisableLog("rdApp.*")

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402


MAX_ROWS = 3000
SIM_THR = 0.4
QED_THR = 0.5
SA_NORM_THR = 5.0 / 9.0

HIT_THR_BY_TARGET: dict[str, float] = {
    "parp1": 10.0,
    "fa7": 8.5,
    "5ht1b": 8.7845,
    "braf": 10.3,
    "jak2": 9.1,
}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ZINC250K_CSV = str((BASE_DIR / "tool" / "zinc250k.csv").resolve())
DEFAULT_VALID_IDX = str((BASE_DIR / "tool" / "valid_idx_zinc250k.json").resolve())
DEFAULT_NOVEL_CACHE = str((BASE_DIR / "tool" / "zinc250k_novelty.pt").resolve())


def _safe_divide(n_parts: int, seq: list) -> list[list]:
    if n_parts <= 1:
        return [list(seq)]
    if mit is not None:
        return [list(c) for c in mit.divide(n_parts, seq)]

    out: list[list] = []
    total = len(seq)
    q, r = divmod(total, n_parts)
    start = 0
    for i in range(n_parts):
        size = q + (1 if i < r else 0)
        out.append(seq[start : start + size])
        start += size
    return out


def _load_or_build_novelty_cache(
    zinc250k_csv: Path,
    valid_idx_json: Path,
    novelty_cache: Path,
) -> tuple[set[str], list]:
    if novelty_cache.exists():
        try:
            # PyTorch >= 2.6 defaults weights_only=True; RDKit bitvectors require full pickle load.
            train_smiles, train_fps = torch.load(str(novelty_cache), weights_only=False)
        except TypeError:
            # Backward compatibility for older PyTorch without the weights_only argument.
            train_smiles, train_fps = torch.load(str(novelty_cache))
        return train_smiles, train_fps

    if not zinc250k_csv.exists():
        raise FileNotFoundError(f"zinc250k csv not found: {zinc250k_csv}")
    if not valid_idx_json.exists():
        raise FileNotFoundError(f"valid_idx_zinc250k.json not found: {valid_idx_json}")

    print("Preprocessing ZINC250k for novelty calculation")
    df = pd.read_csv(zinc250k_csv)
    if "smiles" not in df.columns:
        raise ValueError(f"zinc250k csv must contain column 'smiles', got: {list(df.columns)}")

    with valid_idx_json.open("r", encoding="utf-8") as f:
        test_idx = set(json.load(f))
    train_idx = [i for i in range(len(df)) if i not in test_idx]

    train_smiles_series = df.iloc[train_idx]["smiles"]
    train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles_series]
    train_smiles = set(
        [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in train_mols if mol is not None]
    )
    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in train_mols if mol is not None
    ]
    novelty_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save((train_smiles, train_fps), str(novelty_cache))
    return train_smiles, train_fps


def reward_qed(mols: list[Chem.Mol]) -> list[float]:
    return [float(QED.qed(m)) for m in mols]


def reward_sa(mols: list[Chem.Mol]) -> list[float]:
    return [(10.0 - float(sascorer.calculateScore(m))) / 9.0 for m in mols]


def get_novelty(df: pd.DataFrame, train_fps: list) -> None:
    if "FPS" not in df:
        df["FPS"] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df["MOL"]]

    max_sims = []
    for fps in df["FPS"]:
        max_sim = max(DataStructs.BulkTanimotoSimilarity(fps, train_fps))
        max_sims.append(max_sim)
    df["SIM"] = max_sims


def similarity_matrix_tanimoto(fps1: list, fps2: list) -> np.ndarray:
    similarities = [DataStructs.BulkTanimotoSimilarity(fp, fps2) for fp in fps1]
    return np.array(similarities)


class NCircles:
    def __init__(self, threshold: float = 0.75):
        self.sim_mat_func = similarity_matrix_tanimoto
        self.t = threshold

    def get_circles(self, args) -> list:
        vecs, sim_mat_func, t = args
        circs = []
        for vec in vecs:
            if len(circs) > 0:
                dists = 1.0 - sim_mat_func([vec], circs)
                if dists.min() <= t:
                    continue
            circs.append(vec)
        return circs

    def measure(self, vecs: list, n_chunk: int = 64) -> int:
        for i in range(3):
            vecs_list = _safe_divide(max(1, n_chunk // (2**i)), vecs)
            args = zip(
                vecs_list,
                [self.sim_mat_func] * len(vecs_list),
                [self.t] * len(vecs_list),
            )
            circs_list = list(map(self.get_circles, args))
            vecs = [c for ls in circs_list for c in ls]
            random.shuffle(vecs)
        vecs = self.get_circles((vecs, self.sim_mat_func, self.t))
        return len(vecs)


def get_ncircle(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if "FPS" not in df:
        df["FPS"] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df["MOL"]]
    return NCircles().measure(df["FPS"].tolist())


def get_hit_top5(df: pd.DataFrame, n_total_smi: int, hit_thr: float) -> tuple[float, float]:
    hit_ratio = len(df[df["DOCKING"] > hit_thr]) / n_total_smi
    idx_tmp = int(n_total_smi * 0.05)
    top_5_score = (
        df.sort_values(by="DOCKING", ascending=False)["DOCKING"].iloc[:idx_tmp].mean()
        if idx_tmp > 0
        else float("nan")
    )
    return hit_ratio, top_5_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="results/sbdd/softmol/main/parp1_seed44.csv",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="parp1",
        choices=["parp1", "fa7", "5ht1b", "braf", "jak2"],
    )
    parser.add_argument("--zinc250k-csv", type=str, default=DEFAULT_ZINC250K_CSV)
    parser.add_argument("--valid-idx-json", type=str, default=DEFAULT_VALID_IDX)
    parser.add_argument("--novel-cache", type=str, default=DEFAULT_NOVEL_CACHE)
    parser.add_argument("--sim-thr", type=float, default=SIM_THR)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    zinc250k_csv = Path(args.zinc250k_csv).expanduser().resolve()
    valid_idx_json = Path(args.valid_idx_json).expanduser().resolve()
    novelty_cache = Path(args.novel_cache).expanduser().resolve()

    hit_thr = HIT_THR_BY_TARGET[args.target]

    df = pd.read_csv(input_path).iloc[:MAX_ROWS].copy()
    n_total_smi = len(df)
    print(f"Number of molecules:\t{n_total_smi}")
    if n_total_smi == 0:
        return

    if "smi" not in df.columns or "rv" not in df.columns:
        raise ValueError(f"Input CSV must contain ['smi', 'rv']; got {list(df.columns)}")

    df["SMILES"] = df["smi"].astype(str)
    df["DOCKING"] = pd.to_numeric(df["rv"], errors="coerce")
    df["MOL"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["MOL"]).copy()

    _, train_fps = _load_or_build_novelty_cache(
        zinc250k_csv=zinc250k_csv,
        valid_idx_json=valid_idx_json,
        novelty_cache=novelty_cache,
    )
    get_novelty(df, train_fps)
    print(f"Novelty:\t\t{len(df[df['SIM'] < args.sim_thr]) / n_total_smi}")

    df = df.drop_duplicates(subset=["SMILES"]).copy()

    if "QED" not in df:
        df["QED"] = reward_qed(df["MOL"].tolist())
    if "SA" not in df:
        df["SA"] = reward_sa(df["MOL"].tolist())

    df = df[df["QED"] > QED_THR]
    df = df[df["SA"] > SA_NORM_THR]

    df2 = df[df["DOCKING"] > hit_thr].copy()
    ncircle = get_ncircle(df2)
    print(f"#Circle:\t\t{ncircle}")

    df = df[df["SIM"] < args.sim_thr].copy()
    hit_ratio, top_5_score = get_hit_top5(df, n_total_smi=n_total_smi, hit_thr=hit_thr)
    print(f"Novel hit ratio:\t{hit_ratio}")
    print(f"Novel top 5% DS:\t{top_5_score}")


if __name__ == "__main__":
    main()
