from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

from gated_mcts.mcts import MCTSConfig, MolecularProblemState
from gated_mcts.utils.chem_utils import sentence2mol
from gated_mcts.utils.docking.docking_utils import DockingVina


TRACE_FIELDNAMES = [
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


@dataclass
class DockingRecord:
    call_idx: int
    smiles: str
    affinity: float


class DockingBudgetWrapper:
    """Budgeted docking oracle for Greedy/Beam ablations.

    - Budget counts actual docking calls, not unique SMILES.
    - Invalid / gate-filtered molecules do not consume budget because they never reach this wrapper.
    """

    def __init__(self, target: str, max_oracle_calls: int):
        self.oracle = DockingVina(target)
        self.max_oracle_calls = max(0, int(max_oracle_calls))
        self.history: list[DockingRecord] = []

    @property
    def n_calls(self) -> int:
        return len(self.history)

    @property
    def finish(self) -> bool:
        return self.n_calls >= self.max_oracle_calls

    def predict(self, smiles_list: list[str]) -> list[float]:
        smiles_list = list(smiles_list or [])
        if len(smiles_list) == 0:
            return []

        remaining = max(self.max_oracle_calls - self.n_calls, 0)
        if remaining <= 0:
            return [1.0] * len(smiles_list)

        active = smiles_list[:remaining]
        values = [float(v) for v in self.oracle.predict(active)]
        for smi, affinity in zip(active, values):
            self.history.append(
                DockingRecord(
                    call_idx=len(self.history) + 1,
                    smiles=str(smi),
                    affinity=float(affinity),
                )
            )

        if len(smiles_list) > remaining:
            values.extend([1.0] * (len(smiles_list) - remaining))
        return values


@dataclass
class SearchNode:
    state: MolecularProblemState
    node_id: str
    path: tuple[int, ...]
    score: float = 0.0
    rv: float | None = None
    sentence: str | None = None
    smi: str | None = None
    terminal: bool = False


def _rank_key(node: SearchNode) -> tuple[float, float, float]:
    rv = float("-inf") if node.rv is None else float(node.rv)
    return (float(node.score), rv, 1.0 if node.terminal else 0.0)


def write_trace_csv(trace_path: str, search_obj) -> None:
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    events = search_obj.get_trace() if hasattr(search_obj, "get_trace") else []
    with open(trace_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDNAMES)
        writer.writeheader()
        for ev in events:
            row = {k: ev.get(k, "") for k in TRACE_FIELDNAMES}
            if isinstance(row.get("path"), list):
                row["path"] = "[" + ",".join(str(x) for x in row["path"]) + "]"
            writer.writerow(row)


def make_trace_path(base_dir: str, base_name: str, idx: int) -> str:
    name, ext = os.path.splitext(base_name)
    if ext == ".csv":
        final = f"{name}_{idx}{ext}"
    else:
        final = f"{name}_{idx}.csv"
    return os.path.join(base_dir, final)


class BaseBaselineSearcher:
    def __init__(self, initial_state: MolecularProblemState, config: MCTSConfig):
        self.initial_state = initial_state
        self.config = config
        self.root = SearchNode(
            state=initial_state,
            node_id="root",
            path=(),
            terminal=bool(initial_state.is_terminal()),
        )
        self.time_taken = 0.0
        self.max_search_depth = 0
        self.total_steps = 0
        self.total_rollouts = 0
        self.total_requests = 0
        self._trace: list[dict] = []
        self._trace_meta: dict = {}
        self._best_rv: float | None = None
        self._best_smi: str | None = None
        self._best_sentence: str | None = None

    def _log_event(self, ev: dict) -> None:
        try:
            self._trace.append(dict(ev))
        except Exception:
            pass

    def get_trace(self) -> list[dict]:
        return list(self._trace)

    def get_trace_meta(self) -> dict:
        return dict(self._trace_meta)

    def get_time(self) -> float:
        return float(self.time_taken)

    def get_max_search_depth(self) -> int:
        return int(self.max_search_depth)

    def _oracle_calls(self) -> int:
        predictor = getattr(self.initial_state, "predictor", None)
        if predictor is None or not hasattr(predictor, "n_calls"):
            return 0
        return int(getattr(predictor, "n_calls"))

    def _predictor_finished(self) -> bool:
        predictor = getattr(self.initial_state, "predictor", None)
        if predictor is None or not hasattr(predictor, "finish"):
            return False
        return bool(getattr(predictor, "finish"))

    def _remaining_budget(self) -> int:
        return max(0, int(getattr(self.config, "search_time", 0)) - self._oracle_calls())

    def _current_width(self, depth: int) -> int:
        if depth <= 0:
            init_children = int(getattr(self.config, "init_children", -1))
            if init_children > -1:
                return max(1, init_children)
        return max(1, int(getattr(self.config, "n_total_children", 1)))

    def _sample_k(self, width: int) -> int:
        eval_bsz = int(getattr(self.initial_state.model, "eval_bsz", 1))
        return max(1, eval_bsz, int(width))

    def _maybe_update_best(self, rv: float | None, smi: str | None, sentence: str | None) -> None:
        if rv is None or smi is None or sentence is None:
            return
        if self._best_rv is None or float(rv) > float(self._best_rv):
            self._best_rv = float(rv)
            self._best_smi = smi
            self._best_sentence = sentence

    def _truncate_one_block_suffix(self, suf_ids: list[int]) -> list[int]:
        block_sz = int(getattr(self.initial_state.model, "block_size", 4))
        eos_id = int(self.initial_state.tokenizer.eos_token_id)
        window = list(suf_ids[:block_sz])
        if eos_id in window:
            idx = window.index(eos_id)
            return window[: idx + 1]
        return window

    def _collect_unique_children(
        self,
        state: MolecularProblemState,
        seen_actions: set[tuple[int, ...]],
        sample_k: int,
    ) -> list[list[int]]:
        if state.is_terminal():
            return []

        prefix_ids = state.cur_molecule[0].tolist()
        prefix_text = state.cur_sentence.replace("[BOS]", "").replace("[EOS]", "")
        texts = state.model.generate_block(prefix=prefix_text, k=max(1, int(sample_k)))
        ids_prefix = state.tokenizer.encode(prefix_text, add_special_tokens=False)

        actions: list[list[int]] = []
        for text in texts:
            ids_full = state.tokenizer.encode(text, add_special_tokens=False)
            suffix_ids = ids_full[len(ids_prefix):]
            suffix_ids = self._truncate_one_block_suffix(suffix_ids)
            if len(suffix_ids) == 0:
                continue
            action = tuple(prefix_ids + list(suffix_ids))
            if action in seen_actions:
                continue
            seen_actions.add(action)
            actions.append(list(action))
        return actions

    def _rollout_once(self, state: MolecularProblemState) -> tuple[float, float, str | None, str]:
        action, sentence, _ = state.generate_fragment(
            cur_molecule=state.cur_molecule,
            is_simulation=True,
        )
        _ = action
        _, smiles = sentence2mol(sentence, True)
        if self._predictor_finished():
            return 0.0, 0.0, smiles, sentence
        rv, reward = state.get_reward(smiles)
        return float(rv), float(reward), smiles, sentence

    def _score_node(self, node: SearchNode) -> None:
        reward = 0.0
        best_rv: float | None = None
        best_sentence: str | None = None
        best_smi: str | None = None
        rollouts_used = 0

        if float(getattr(self.config, "value_weight", 0.0)) > 0 and not self._predictor_finished():
            rv_value, raw_value = node.state.get_value()
            reward += float(self.config.value_weight) * float(raw_value)
            if node.state.is_terminal():
                best_rv = float(rv_value)
                best_sentence = node.state.cur_sentence
                _, best_smi = sentence2mol(best_sentence, True)

        if node.state.is_terminal():
            if float(getattr(self.config, "fastrollout_weight", 0.0)) > 0 and not self._predictor_finished():
                rv_term, raw_term = node.state.get_value()
                reward += float(self.config.fastrollout_weight) * float(raw_term)
                best_rv = float(rv_term)
                best_sentence = node.state.cur_sentence
                _, best_smi = sentence2mol(best_sentence, True)
        elif (
            int(getattr(self.config, "n_simulations", 0)) > 0
            and float(getattr(self.config, "fastrollout_weight", 0.0)) > 0
        ):
            rollout_vals: list[tuple[float, float, str | None, str]] = []
            for _ in range(max(1, int(self.config.n_simulations))):
                if self._predictor_finished():
                    break
                rollout_vals.append(self._rollout_once(node.state))
                rollouts_used += 1
            if len(rollout_vals) > 0:
                best_rollout = max(rollout_vals, key=lambda x: (float(x[1]), float(x[0])))
                best_rv = float(best_rollout[0])
                best_smi = best_rollout[2]
                best_sentence = best_rollout[3]
                reward += float(self.config.fastrollout_weight) * float(best_rollout[1])

        node.score = float(reward)
        node.rv = None if best_rv is None else float(best_rv)
        node.sentence = best_sentence
        node.smi = best_smi
        node.terminal = bool(node.state.is_terminal())

        self.total_steps += 1
        self.total_rollouts += int(rollouts_used)
        self.max_search_depth = max(self.max_search_depth, int(node.state.cur_step))
        self._maybe_update_best(node.rv, node.smi, node.sentence)

        self._log_event(
            {
                "iter": int(self.total_steps),
                "type": "score",
                "node_id": node.node_id,
                "depth": int(node.state.cur_step),
                "path": list(node.path),
                "rv": None if node.rv is None else float(node.rv),
                "reward": float(node.score),
                "n_visits": 1,
                "q_sum": float(node.score),
                "n_children": 0,
                "terminal": bool(node.terminal),
                "sentence": node.state.cur_sentence,
                "oracle_calls": int(self._oracle_calls()),
            }
        )

    def _finalize(self) -> tuple[float | None, str | None, str | None]:
        self._log_event(
            {
                "type": "final",
                "rv": None if self._best_rv is None else float(self._best_rv),
                "smi": self._best_smi,
                "sentence": self._best_sentence,
                "total_steps": int(self.total_steps),
                "total_rollouts": int(self.total_rollouts),
                "total_requests": int(self.total_requests),
                "oracle_calls": int(self._oracle_calls()),
            }
        )
        return self._best_rv, self._best_smi, self._best_sentence

    def run_search(self) -> tuple[float | None, str | None, str | None]:
        raise NotImplementedError

    def run(self) -> tuple[float | None, str | None, str | None]:
        start_time = time.time()
        rv, smi, sentence = self.run_search()
        self.time_taken = time.time() - start_time
        print(f"run_time:{self.time_taken / 60 :.2f}min")
        return rv, smi, sentence


class GreedySearcher(BaseBaselineSearcher):
    def run_search(self) -> tuple[float | None, str | None, str | None]:
        max_budget = int(getattr(self.config, "search_time", 0))
        self._trace_meta = {
            "algorithm": "greedy",
            "oracle_budget": max_budget,
            "search_time": max_budget,
            "init_children": int(getattr(self.config, "init_children", -1)),
            "n_total_children": int(getattr(self.config, "n_total_children", -1)),
            "n_simulations": int(getattr(self.config, "n_simulations", 0)),
            "fastrollout_weight": float(getattr(self.config, "fastrollout_weight", 0.0)),
            "ts_start": time.time(),
        }

        current = self.root
        stagnation_limit = max(1, int(getattr(self.config, "max_resample_on_empty", 5)))
        pbar = tqdm(total=max_budget, desc="Greedy oracle calls", leave=True)
        last_calls = 0

        while not current.state.is_terminal() and not self._predictor_finished():
            width = self._current_width(int(current.state.cur_step))
            sample_k = self._sample_k(width)
            seen_actions: set[tuple[int, ...]] = set()
            candidates: list[SearchNode] = []
            no_new_rounds = 0
            no_oracle_progress_rounds = 0

            while not self._predictor_finished():
                round_calls_before = self._oracle_calls()
                actions = self._collect_unique_children(current.state, seen_actions, sample_k)
                self.total_requests += 1
                if len(actions) == 0:
                    no_new_rounds += 1
                    if no_new_rounds >= stagnation_limit:
                        self._log_event(
                            {
                                "iter": int(self.total_steps),
                                "type": "stop_exhausted",
                                "node_id": current.node_id,
                                "depth": int(current.state.cur_step),
                                "path": list(current.path),
                                "oracle_calls": int(self._oracle_calls()),
                            }
                        )
                        break
                    continue

                no_new_rounds = 0
                for action in actions:
                    if self._predictor_finished():
                        break
                    local_idx = len(candidates)
                    child = SearchNode(
                        state=current.state.take_action(action),
                        node_id=f"{current.node_id}-{local_idx}",
                        path=current.path + (local_idx,),
                    )
                    self._score_node(child)
                    candidates.append(child)

                    now_calls = self._oracle_calls()
                    if now_calls > last_calls:
                        pbar.update(now_calls - last_calls)
                        last_calls = now_calls

                if self._oracle_calls() == round_calls_before:
                    no_oracle_progress_rounds += 1
                    if no_oracle_progress_rounds >= stagnation_limit:
                        self._log_event(
                            {
                                "iter": int(self.total_steps),
                                "type": "stop_no_oracle_progress",
                                "node_id": current.node_id,
                                "depth": int(current.state.cur_step),
                                "path": list(current.path),
                                "oracle_calls": int(self._oracle_calls()),
                            }
                        )
                        break
                else:
                    no_oracle_progress_rounds = 0

            if len(candidates) == 0:
                break

            best_child = max(candidates, key=_rank_key)
            current = best_child
            self._log_event(
                {
                    "iter": int(self.total_steps),
                    "type": "advance",
                    "node_id": current.node_id,
                    "depth": int(current.state.cur_step),
                    "path": list(current.path),
                    "rv": None if current.rv is None else float(current.rv),
                    "reward": float(current.score),
                    "terminal": bool(current.terminal),
                    "sentence": current.state.cur_sentence,
                    "oracle_calls": int(self._oracle_calls()),
                }
            )

        if self._predictor_finished():
            self._log_event(
                {
                    "iter": int(self.total_steps),
                    "type": "stop_budget",
                    "node_id": current.node_id,
                    "depth": int(current.state.cur_step),
                    "path": list(current.path),
                    "oracle_calls": int(self._oracle_calls()),
                }
            )

        now_calls = self._oracle_calls()
        if now_calls > last_calls:
            pbar.update(now_calls - last_calls)
        pbar.close()
        return self._finalize()


class BeamSearcher(BaseBaselineSearcher):
    def run_search(self) -> tuple[float | None, str | None, str | None]:
        max_budget = int(getattr(self.config, "search_time", 0))
        self._trace_meta = {
            "algorithm": "beam",
            "oracle_budget": max_budget,
            "search_time": max_budget,
            "init_children": int(getattr(self.config, "init_children", -1)),
            "n_total_children": int(getattr(self.config, "n_total_children", -1)),
            "n_simulations": int(getattr(self.config, "n_simulations", 0)),
            "fastrollout_weight": float(getattr(self.config, "fastrollout_weight", 0.0)),
            "ts_start": time.time(),
        }

        beam: list[SearchNode] = [self.root]
        layer_idx = 0
        stagnation_limit = max(1, int(getattr(self.config, "max_resample_on_empty", 5)))
        pbar = tqdm(total=max_budget, desc="Beam oracle calls", leave=True)
        last_calls = 0

        while len(beam) > 0 and not self._predictor_finished():
            width = self._current_width(layer_idx)
            sample_k = self._sample_k(width)
            no_new_rounds = 0
            no_oracle_progress_rounds = 0
            layer_candidates: list[SearchNode] = []
            layer_seen_actions: set[tuple[int, ...]] = set()

            while not self._predictor_finished():
                round_calls_before = self._oracle_calls()
                round_progress = False
                for parent in beam:
                    if parent.state.is_terminal() or self._predictor_finished():
                        continue
                    actions = self._collect_unique_children(parent.state, layer_seen_actions, sample_k)
                    self.total_requests += 1
                    if len(actions) == 0:
                        continue
                    round_progress = True
                    for action in actions:
                        if self._predictor_finished():
                            break
                        child = SearchNode(
                            state=parent.state.take_action(action),
                            node_id=f"{parent.node_id}-{len(layer_candidates)}",
                            path=parent.path + (len(layer_candidates),),
                        )
                        self._score_node(child)
                        layer_candidates.append(child)

                        now_calls = self._oracle_calls()
                        if now_calls > last_calls:
                            pbar.update(now_calls - last_calls)
                            last_calls = now_calls

                if round_progress:
                    no_new_rounds = 0
                else:
                    no_new_rounds += 1
                    if no_new_rounds >= stagnation_limit:
                        self._log_event(
                            {
                                "iter": int(self.total_steps),
                                "type": "stop_exhausted",
                                "depth": int(layer_idx),
                                "oracle_calls": int(self._oracle_calls()),
                            }
                        )
                        break

                if self._oracle_calls() == round_calls_before:
                    no_oracle_progress_rounds += 1
                    if no_oracle_progress_rounds >= stagnation_limit:
                        self._log_event(
                            {
                                "iter": int(self.total_steps),
                                "type": "stop_no_oracle_progress",
                                "depth": int(layer_idx),
                                "oracle_calls": int(self._oracle_calls()),
                            }
                        )
                        break
                else:
                    no_oracle_progress_rounds = 0

            if len(layer_candidates) == 0:
                break

            layer_candidates.sort(key=_rank_key, reverse=True)
            beam = layer_candidates[:width]
            for rank, node in enumerate(beam):
                self._log_event(
                    {
                        "iter": int(self.total_steps),
                        "type": "keep",
                        "node_id": node.node_id,
                        "depth": int(node.state.cur_step),
                        "path": list(node.path),
                        "rv": None if node.rv is None else float(node.rv),
                        "reward": float(node.score),
                        "terminal": bool(node.terminal),
                        "sentence": node.state.cur_sentence,
                        "oracle_calls": int(self._oracle_calls()),
                        "n_children": int(rank),
                    }
                )
            layer_idx += 1

        if self._predictor_finished():
            self._log_event(
                {
                    "iter": int(self.total_steps),
                    "type": "stop_budget",
                    "depth": int(layer_idx),
                    "oracle_calls": int(self._oracle_calls()),
                }
            )

        now_calls = self._oracle_calls()
        if now_calls > last_calls:
            pbar.update(now_calls - last_calls)
        pbar.close()
        return self._finalize()
