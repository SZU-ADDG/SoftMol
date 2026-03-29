from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, List, Tuple

from rdkit import Chem, rdBase
from rdkit.Chem import QED

from gated_mcts.utils import sascorer

rdBase.DisableLog("rdApp.error")


LOCAL_PROPERTY_ORACLES: Tuple[str, ...] = ("qed", "sa")


@dataclass
class LocalPropertyRecord:
    call_idx: int
    smiles: str
    score: float


class LocalPropertyOracle:
    """Local property oracle with PMO-compatible budget counting."""

    def __init__(self, oracle_name: str, max_oracle_calls: int = 10000, freq_log: int = 100):
        if oracle_name not in LOCAL_PROPERTY_ORACLES:
            raise ValueError(f"Unsupported oracle '{oracle_name}'.")
        self.oracle_name = str(oracle_name)
        self.max_oracle_calls = int(max_oracle_calls)
        self.freq_log = int(freq_log)

        # canonical_smi -> (score, first_seen_idx)
        self.buffer: Dict[str, Tuple[float, int]] = {}
        self.history: List[LocalPropertyRecord] = []

    @staticmethod
    def canonicalize(smiles: str | None) -> str | None:
        if smiles is None:
            return None
        smi = str(smiles).strip()
        if len(smi) == 0:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    @property
    def n_calls(self) -> int:
        return len(self.buffer)

    @property
    def finish(self) -> bool:
        return self.n_calls >= self.max_oracle_calls

    def _score_mol(self, mol) -> float:
        if self.oracle_name == "qed":
            score = float(QED.qed(mol))
        elif self.oracle_name == "sa":
            sa_raw = float(sascorer.calculateScore(mol))
            score = float((10.0 - sa_raw) / 9.0)
        else:
            raise ValueError(f"Unsupported oracle '{self.oracle_name}'.")

        if not isfinite(score):
            return 0.0
        return max(0.0, score)

    def score_smiles(self, smiles: str | None) -> float:
        cano = self.canonicalize(smiles)
        if cano is None:
            return 0.0

        if cano in self.buffer:
            return float(self.buffer[cano][0])

        if self.finish:
            return 0.0

        mol = Chem.MolFromSmiles(cano)
        if mol is None:
            return 0.0

        try:
            score = self._score_mol(mol)
        except Exception:
            score = 0.0

        if not isfinite(score):
            score = 0.0

        call_idx = self.n_calls + 1
        self.buffer[cano] = (float(score), call_idx)
        self.history.append(LocalPropertyRecord(call_idx=call_idx, smiles=cano, score=float(score)))
        return float(score)

    def predict(self, smiles_list: List[str]) -> List[float]:
        return [self.score_smiles(s) for s in smiles_list]

    def topk(self, k: int = 100) -> List[LocalPropertyRecord]:
        k = max(1, int(k))
        rows = [LocalPropertyRecord(call_idx=v[1], smiles=s, score=v[0]) for s, v in self.buffer.items()]
        rows.sort(key=lambda x: (-x.score, x.call_idx))
        return rows[:k]
