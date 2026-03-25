from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from rdkit import Chem
from tdc import Oracle


PMO_ORACLES: Tuple[str, ...] = (
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
)


@dataclass
class PMORecord:
    call_idx: int
    smiles: str
    score: float


class PMOOracleAdapter:
    """PMO oracle adapter with GenMol-compatible budget counting rules.

    - Budget counts unique canonical SMILES only.
    - Invalid SMILES return score 0.0 and do not consume budget.
    - Duplicate canonical SMILES reuse cached scores and do not consume budget.
    """

    def __init__(self, oracle_name: str, max_oracle_calls: int = 10000, freq_log: int = 100):
        if oracle_name not in PMO_ORACLES:
            raise ValueError(f"Unsupported oracle '{oracle_name}'.")
        self.oracle_name = oracle_name
        self.max_oracle_calls = int(max_oracle_calls)
        self.freq_log = int(freq_log)
        self.oracle = Oracle(name=oracle_name)

        # canonical_smi -> (score, first_seen_idx)
        self.buffer: Dict[str, Tuple[float, int]] = {}
        self.history: List[PMORecord] = []

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

    def score_smiles(self, smiles: str | None) -> float:
        cano = self.canonicalize(smiles)
        if cano is None:
            return 0.0

        if cano in self.buffer:
            return float(self.buffer[cano][0])

        if self.finish:
            return 0.0

        try:
            raw_score = self.oracle(cano)
            score = float(raw_score)
        except Exception:
            score = 0.0

        call_idx = self.n_calls + 1
        self.buffer[cano] = (score, call_idx)
        self.history.append(PMORecord(call_idx=call_idx, smiles=cano, score=score))
        return score

    def predict(self, smiles_list: List[str]) -> List[float]:
        return [self.score_smiles(s) for s in smiles_list]

    def topk(self, k: int = 100) -> List[PMORecord]:
        k = max(1, int(k))
        rows = [PMORecord(call_idx=v[1], smiles=s, score=v[0]) for s, v in self.buffer.items()]
        rows.sort(key=lambda x: (-x.score, x.call_idx))
        return rows[:k]
