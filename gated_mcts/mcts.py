from __future__ import annotations

import re
import os
import sys
from pathlib import Path
from gated_mcts.utils.chem_utils import (
    sentence2mol,
)
import time
import numpy as np
from collections import deque
from rdkit import Chem
import torch
from tqdm import tqdm
import pickle

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import omegaconf
import dataloader
import diffusion


class BD3Sampler:
    """Persistent sampler: loads tokenizer and Diffusion in-process, loads once for multiple samples.

    - generate_block: generates only "one block after prefix" (depends on next_block_only).
    - generate_full: generates from given prefix to EOS (used by Simulation).
    """

    def __init__(
        self,
        *,
        gpu: str,
        ckpt: str,
        vocab: str,
        length: int,
        block_size: int,
        steps: int = 128,
        nucleus: float = 0.95,
        eval_bsz: int = 64,
        model_name: str = 'small',
        out: str = './sample_logs/mcts_temp',
        seed: int = 42,
        temperature: float = 1.0,
    ) -> None:
        self.gpu = str(gpu)
        self.ckpt = str(ckpt)
        self.vocab = str(vocab)
        self.length = int(length)
        self.block_size = int(block_size)
        self.steps = int(steps)
        self.nucleus = float(nucleus)
        self.eval_bsz = int(eval_bsz)
        self.model_name = str(model_name)
        self.out = str(out)
        self.temperature = float(temperature)

        base_cfg = omegaconf.OmegaConf.load("configs/sample.yaml")
        model_yaml = Path("configs/model") / f"{self.model_name}.yaml"
        model_cfg = omegaconf.OmegaConf.load(str(model_yaml))
        overrides = omegaconf.OmegaConf.create(
            {
                "seed": int(seed),
                "block_size": int(self.block_size),
                "algo": {"T": int(self.steps)},
                "model": {"length": int(self.length), "attn_backend": "sdpa"},
                "loader": {"eval_batch_size": int(self.eval_bsz)},
                "sampling": {
                    "logdir": str(Path(self.out).resolve()),
                    "nucleus_p": float(self.nucleus),
                    "prefix": "",
                    "next_block_only": False,
                    "temperature": float(self.temperature),
                },
                "eval": {"checkpoint_path": str(Path(self.ckpt).resolve())},
                "data": {"tokenizer_name_or_path": str(Path(self.vocab).resolve())},
            }
        )
        self.config = omegaconf.OmegaConf.merge(base_cfg, {"model": model_cfg}, overrides)

        # Persistent Tokenizer and Diffusion
        self.tokenizer_runtime = dataloader.get_tokenizer(self.config)
        self.mdl = diffusion.Diffusion.load_from_checkpoint(
            self.config.eval.checkpoint_path,
            tokenizer=self.tokenizer_runtime,
            config=self.config,
            strict=False,
            weights_only=False,
        ).to("cuda")
        if self.config.eval.disable_ema:
            self.mdl.ema = None
        self.mdl.backbone.eval()
        self.mdl.noise.eval()

    def _clean_text_remove_bos(self, samples: list[str]) -> list[str]:
        """Unified cleaning function: removes only [BOS], keeps [EOS] for termination check.

        - Applicable to Expansion (next block only) and Simulation (generate to EOS).
        - Unified behavior ensures MCTS correctly perceives termination when model generates [EOS] within a block.
        """
        drop_tokens = ["[BOS]"]
        cleaned = []
        for s in samples:
            t = s
            for tk in drop_tokens:
                t = t.replace(tk, "")
            cleaned.append(t)
        return cleaned

    def _set_runtime_sampling(self, *, k: int, prefix: str | None, next_block_only: bool):
        # Dynamically update runtime sampling config (does not rebuild model)
        self.mdl.config.loader.eval_batch_size = int(k)
        self.mdl.config.sampling.nucleus_p = float(self.nucleus)
        self.mdl.config.sampling.temperature = float(self.temperature)
        self.mdl.config.sampling.prefix = prefix or ""
        self.mdl.config.sampling.next_block_only = bool(next_block_only)

    def generate_block(self, prefix: str, k: int | None = None) -> list[str]:
        """Generate only "one block after prefix", return k candidate texts (no [BOS], keep [EOS])."""
        k = int(k or self.eval_bsz)
        self._set_runtime_sampling(k=k, prefix=prefix, next_block_only=True)
        texts = self.mdl.restore_model_and_sample(num_steps=self.mdl.config.algo.T)
        return self._clean_text_remove_bos(texts)

    def generate_full(self, prefix: str, k: int | None = None) -> list[str]:
        """Generate from prefix to EOS, return k candidate texts (no [BOS], keep [EOS])."""
        k = int(k or self.eval_bsz)
        self._set_runtime_sampling(k=k, prefix=prefix, next_block_only=False)
        texts = self.mdl.restore_model_and_sample(num_steps=self.mdl.config.algo.T)
        return self._clean_text_remove_bos(texts)

class MCTSConfig:

    # optimization parameters
    value_weight = 0            # weight of value in the total reward. 0 means no value.
    search_time = 1000          # total search times (equal or larger than than the number of nodes expanded)
    min_terminals = -1          # minimum number of terminals nodes must be found before stopping, -1 means no force.
    max_split_depth = 10        # maximum depth to split the tree. If larger, only single path will be expanded. If -1, no limit. This is a piror knowledge of the problem.
    init_children = 20          # initial number of children to expand at the root node. if -1, use N_TOTAL_CHILDREN. This is a piror knowledge of the problem. 
    n_total_children = 8        # * number of children to expand at each node
    c_param = 5                 # exploration parameter. Larger means more Exploration.
    width_increase_factor = 2   # increase the width of the tree by this factor in Adaptive child allocation

    add_value_weight = 0.0
    n_simulations = 1
    fastrollout_weight = 1.0

    greedy_path = False
    max_n_repeat = 5

    diversity_threshold = 0.6   # Compatibility: historical similarity threshold, currently only used for "sibling not identical" deduplication
    max_resample_on_empty = 5   # Max resampling attempts when candidate pool is filtered empty

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MolecularProblemState:

    def __init__(self,
                 model,
                 tokenizer,
                 predictor,
                 enable_qed_sa_gate: bool = True,
                 cur_molecule=None, 
                 cur_step=0, 
                 max_steps=10,  
                 is_terminate=False,  
                 rewards=None,  
                 has_optimized=False):  

        self.predictor = predictor
        self.enable_qed_sa_gate = bool(enable_qed_sa_gate)
        self.cur_molecule = cur_molecule
        self.model = model
        self.tokenizer = tokenizer
        sentence = self.tokenizer.decode(self.cur_molecule[0])
        self.cur_sentence = sentence
        self.cur_step = cur_step
        self.max_steps = max_steps
        self.is_terminate = is_terminate
        self.rewards = rewards if rewards is not None else []
        self.has_optimized = has_optimized

    def get_cur_molecule(self):
        return self.cur_molecule

    def get_cur_step(self):
        return self.cur_step

    def is_terminal(self):
        has_eos = self.check_eos_exist()
        max_lines_reached = self.cur_step >= self.max_steps
        return has_eos or max_lines_reached or self.is_terminate

    def check_eos_exist(self):
        if "[EOS]" in self.cur_sentence:
            return True
        else:
            return False

    @staticmethod
    def extract_smiles(completion):
        SMILES_RE = re.compile(r"(?:SMILES:\s*)([A-Za-z0-9@+\-\[\]\(\)=#$%]+)")
        match = SMILES_RE.search(completion)
        if match:
            return match.group(1).strip()
        else:
            return "<INVALID_SMILES>"

    def is_correct(self):
        predicted_smiles = self.extract_smiles(self.cur_molecule)
        if predicted_smiles == "<INVALID_SMILES>":
            return False
        return predicted_smiles

    def get_value(self):
        # return (rv, reward)
        _, smiles = sentence2mol(self.cur_sentence, True)
        rv, value = self.get_reward(smiles)
        return rv, value
    
    def get_reward(self, smiles):
        """Reward based solely on Vina docking score.

        - If smiles is invalid or docking fails/returns non-negative energy, rv = -1.0, reward = -1.0.
        - Otherwise rv = -affinity (lower energy is better, larger rv is better), reward = rv.
        """
        if smiles is None:
            return -1.0, -1.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1.0, -1.0

        # QED/SA gate (enabled by default): Docking and scoring are performed only when QED > 0.5 and SA_raw < 5.0.
        if self.enable_qed_sa_gate:
            from rdkit.Chem import QED
            from gated_mcts.utils import sascorer as _sascorer

            qed = float(QED.qed(mol))
            sa_raw = float(_sascorer.calculateScore(mol))  # Raw SA (lower is better, range approx 1~10)
            if not (qed > 0.5 and sa_raw < 5.0):
                return -1.0, -1.0

        try:
            result = self.predictor.predict([smiles])
            affinity = result[0] if len(result) > 0 else 1.0
        except Exception:
            return -1.0, -1.0

        # Docking energy is usually negative, more negative is better; reward is its negation (larger is better)
        if affinity >= 0:
            return -1.0, -1.0
        rv = -float(affinity)
        reward = rv
        return rv, reward

    def cond_actions(self, to_end=False, is_greedy=False, *, sibling_suffixes=None, sim_threshold: float | None = None, max_resample: int | None = None):
        n_attempts = 5
        for attempt in range(n_attempts):
            try:
                if to_end:
                    action, smiles_answer, has_end_token = self.action2end(
                        is_greedy=is_greedy,
                        sibling_suffixes=sibling_suffixes,
                        sim_threshold=sim_threshold,
                        max_resample=max_resample,
                    )
                else:
                    action, smiles_answer, has_end_token = self.actions(
                        is_greedy=is_greedy,
                        sibling_suffixes=sibling_suffixes,
                        sim_threshold=sim_threshold,
                        max_resample=max_resample,
                    )
                    if len(action) == 0:
                        continue
                return action, smiles_answer, has_end_token
            except Exception as e:
                if attempt < n_attempts - 1:
                    print(f'Retry {attempt}, error: {type(e).__name__}', flush=True)
                    continue
                else:
                    raise e

    def actions(self, is_greedy=False, *, sibling_suffixes=None, sim_threshold: float | None = None, max_resample: int | None = None):
        action, smiles_answer, has_end_token = self.generate_fragment(
            cur_molecule=self.cur_molecule,
            is_simulation=False,  # False means generate only one fragment (Expansion)
            sibling_suffixes=sibling_suffixes,
            sim_threshold=sim_threshold,
            max_resample=(max_resample or 0),
        )
        return action, smiles_answer, has_end_token

    def take_action(self, action):
        new_answer = torch.as_tensor(action, dtype=self.cur_molecule.dtype, device=self.cur_molecule.device).unsqueeze(0)
        next_state = MolecularProblemState(
            model=self.model,
            tokenizer=self.tokenizer,
            predictor=self.predictor,
            enable_qed_sa_gate=self.enable_qed_sa_gate,
            cur_molecule=new_answer,
            cur_step=self.cur_step + 1,
            max_steps=self.max_steps,
            is_terminate=False  
        )
        return next_state

    def action2end(self, is_greedy, *, sibling_suffixes=None, sim_threshold: float | None = None, max_resample: int | None = None):
        action, smiles_answer, has_end_token = self.generate_fragment(
            cur_molecule=self.cur_molecule,
            is_simulation=True,
            sibling_suffixes=sibling_suffixes,
            sim_threshold=sim_threshold,
            max_resample=(max_resample or 0),
        )

        return action, smiles_answer, has_end_token

    def take_action_end(self, is_greedy=False):
        assert is_greedy == False
        if self.is_terminal():
            return self

        n_attempts = 20 
        final_action = ""
        for attempt in range(n_attempts):
            try:
                final_action, smiles_answer, has_end_token = self.action2end(is_greedy=is_greedy)
                break
            except Exception as e:
                if attempt < n_attempts - 1:
                    print(f"[take_action_end] attempt {attempt}, error: {type(e).__name__}. Retrying...")
                    continue
                else:
                    print(f"[take_action_end] All attempts failed. Error: {type(e).__name__}")
                    raise e
        n_steps = smiles_answer.count('[SEP]')

        answer_updated = torch.as_tensor(final_action, dtype=self.cur_molecule.dtype, device=self.cur_molecule.device).unsqueeze(0)

        end_state = MolecularProblemState(
            model=self.model,
            tokenizer=self.tokenizer,
            predictor=self.predictor,
            enable_qed_sa_gate=self.enable_qed_sa_gate,
            cur_molecule=answer_updated,
            cur_step=self.cur_step + n_steps,
            max_steps=1000, 
            is_terminate=True
        )
        return end_state

    def generate_fragment(self, cur_molecule, is_simulation,
                          *, sibling_suffixes=None, sim_threshold: float | None = None, max_resample: int = 0):
        """Sample using SoftBD:
        - Expansion (is_simulation=False): generate only next block; filter out candidates duplicating sibling new blocks, then pick randomly from remaining (fallback to first if empty).
        - Simulation (is_simulation=True): generate to EOS, post-processing remains same (outer logic).
        Returns: (complete token sequence, decoded string, has EOS).
        """
        # 1) Get current prefix text, remove [BOS]/[EOS] tags (sampling prefix convention excludes special tokens)
        prefix_ids = cur_molecule[0].tolist()
        prefix_text = self.tokenizer.decode(cur_molecule[0])
        prefix_text = prefix_text.replace('[BOS]', '').replace('[EOS]', '')

        # 2) Batch generate candidates (temporary candidate pool for this expansion)
        if is_simulation:
            texts = self.model.generate_full(prefix=prefix_text, k=1)
        else:
            texts = self.model.generate_block(prefix=prefix_text, k=getattr(self.model, 'eval_bsz', 64))

        # 3) Map candidates to "new block" token sequences (no exact deduplication)
        ids_prefix = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        pool = []  # [(suffix_ids_list, full_text)]

        # Helper: In Expansion, truncate suffix to "at most one block; if EOS in first block, truncate to EOS (inclusive)".
        def _truncate_one_block_suffix(suf_ids: list[int]) -> list[int]:
            if is_simulation:
                return list(suf_ids)
            block_sz = int(getattr(self.model, 'block_size', 4))
            eos_id = int(self.tokenizer.eos_token_id)
            window = list(suf_ids[:block_sz])
            # Find EOS in the first block window
            if eos_id in window:
                idx = window.index(eos_id)
                return window[: idx + 1]
            return window

        for t in texts:
            ids_full = self.tokenizer.encode(t, add_special_tokens=False)
            suf = ids_full[len(ids_prefix):]
            suf = _truncate_one_block_suffix(suf)
            if len(suf) == 0:
                continue
            pool.append((suf, t))

        # 3.1) Construct hash of sibling new blocks for "deduplication only" strategy
        sibling_suffix_set = set()
        if sibling_suffixes:
            for suf in sibling_suffixes:
                if suf:
                    sibling_suffix_set.add(tuple(suf))

        # 4) Select a candidate:
        #    - Expansion: Find first candidate "not identical to any sibling suffix"; if all duplicate, take first.
        #    - Simulation: Return the first one.
        if not pool and texts:
            # Extreme case: all filtered, fallback to "new part" of first complete text
            ids_full = self.tokenizer.encode(texts[0], add_special_tokens=False)
            suf = ids_full[len(ids_prefix):]
            suf = _truncate_one_block_suffix(suf)
            pool = [(suf, texts[0])]

        if is_simulation:
            # Generate complete molecule: directly use the first one (or fallback)
            suf_ids, chosen_text = pool[0]
        else:
            # Filter out duplicates with siblings, then randomly pick from remainder
            filtered_pool = []
            for cand in pool:
                cand_suffix = tuple(cand[0])
                if not sibling_suffix_set or cand_suffix not in sibling_suffix_set:
                    filtered_pool.append(cand)
            candidate_pool = filtered_pool if filtered_pool else [pool[0]]
            if len(candidate_pool) == 1:
                chosen_pair = candidate_pool[0]
            else:
                random_idx = int(np.random.choice(len(candidate_pool)))
                chosen_pair = candidate_pool[random_idx]
            suf_ids, chosen_text = chosen_pair

        complete_answer = prefix_ids + list(suf_ids)
        smiles_answer = self.tokenizer.decode(complete_answer)
        # EOS presence determined by sampling result (Simulation path generate_full keeps [EOS])
        has_end_token = ('[EOS]' in smiles_answer)

        return complete_answer, smiles_answer, has_end_token


class MonteCarloTreeSearchNode:
    def __init__(self,
                 state,
                 config,
                 parent=None,
                 parent_action=None,
                 depth=0,
                 node_id=None,
                 n_repeat_by_parent=1):

        self.config = config


        self.state = state
        self.parent = parent
        self.parent_action = parent_action  
        self.children = []
        self._number_of_visits = 0
        self._results = []  

        # Cache rv for this node (docking score converted), None means not set yet
        self._values = None  
        self._cached_reward = 0.  

   
        self.depth = depth
        self.node_id = node_id
        self.n_repeat_by_parent = n_repeat_by_parent
        self.n_repeat = 0

        # Cache: "best sample" encountered during Simulation at this node (max rv)
        self._best_rollout_rv = None  # float | None
        self._best_rollout_smi = None  # str | None
        self._best_rollout_sentence = None  # str | None

        # Handle "infinite depth" (-1) semantics: do not modify global config, use local copy in node
        self._max_split_depth = int(getattr(self.config, 'max_split_depth', -1))
        if self._max_split_depth < 0:
            # Use a large enough value to approximate "infinite", avoiding breaking numerical checks
            self._max_split_depth = 10 ** 9

        if self.depth == 0:
            # Root node children count: if init_children==-1, fallback to n_total_children as per comments
            # Otherwise if kept as -1, len(children) >= -1 is always True, treating root as "full" preventing expansion
            self.n_total_children_adaptive = (
                self.config.init_children
                if self.config.init_children > -1
                else self.config.n_total_children
            )
        elif self.depth > self._max_split_depth:
            self.n_total_children_adaptive = 1
        else:
            self.n_total_children_adaptive = self.config.n_total_children

 
        self.max_q_diff = 0
        self.expandable = True

    def n(self):
        return self._number_of_visits

    def q(self):
        return np.sum(self._results)

    def result(self):
        return self._results

    def is_terminal_node(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        return len(self.children) >= self.n_total_children_adaptive

    def n_children(self):
        return len(self.children)

    def total_number_nodes(self):
        tot_node = 1
        for child in self.children:
            tot_node += child.total_number_nodes()
        return tot_node

    def get_ancestor_child_indices(self):
        indices = []
        current_node = self
        while current_node.parent is not None:
            index = current_node.parent.children.index(current_node)
            indices.append(index)
            current_node = current_node.parent
        return indices[::-1]

    def retrieve_origin_value(self):
        return self._values

    def set_cached_reward(self, rv, raw_value):
        self._values = rv
        self._cached_reward = raw_value

    def get_cached_reward(self):
        return self._cached_reward

    def expand(self):
        action, has_end_token, n_repeat = self.get_acceptable_action()
        self.n_repeat = n_repeat

        next_state = self.state.take_action(action)

        cur_n_children = len(self.children)
        cur_node_id = self.node_id
        child_node = MonteCarloTreeSearchNode(
            state=next_state,
            config=self.config,
            parent=self,
            parent_action=action,
            depth=self.depth + 1,
            node_id=f"{cur_node_id}-{cur_n_children}" if cur_node_id else None,
            n_repeat_by_parent=n_repeat
        )

        self.children.append(child_node)
        return child_node

    def get_acceptable_action(self):
        # Expansion phase: no threshold filtering, directly choose candidate most dissimilar to siblings (Token Hamming similarity)
        to_end = self.config.max_split_depth <= (self.depth + 1)
        is_greedy = self.config.greedy_path and len(self.children) == 0

        # Construct "sibling" new block suffixes (slice based on current parent prefix length)
        sibling_suffixes = []
        try:
            prefix_len = len(self.state.cur_molecule[0].tolist())
            for ch in self.children:
                if not hasattr(ch, 'parent_action') or ch.parent_action is None:
                    continue
                if not isinstance(ch.parent_action, (list, tuple)):
                    continue
                if len(ch.parent_action) <= prefix_len:
                    continue
                bro_suf = list(ch.parent_action[prefix_len:])
                if bro_suf:
                    sibling_suffixes.append(bro_suf)
        except Exception:
            sibling_suffixes = []

        action, smiles_answer, has_end_token = self.state.cond_actions(
            to_end=to_end,
            is_greedy=is_greedy,
            sibling_suffixes=sibling_suffixes,
            sim_threshold=None,
            max_resample=int(getattr(self.config, 'max_resample_on_empty', 5)),
        )
        n_repeat = 1
        return action, has_end_token, n_repeat

    def can_expand(self):
        return not self.is_terminal_node() and not self.is_fully_expanded()

    def has_expandable_descendant(self):
        if not self.expandable:
            return False
        if self.can_expand():
            return True
        for child in self.children:
            if child.has_expandable_descendant():
                return True
        self.expandable = False
        return False

    def best_child(self, alpha=0.5):
        valid_children = []
        for child in self.children:
            if child.has_expandable_descendant():
                valid_children.append(child)

        if not valid_children:
            return None

        choices_weights = []
        for c in valid_children:
            exploit = alpha * c.q() / c.n() + (1 - alpha) * max(c.result())
            explore = np.sqrt(np.log(self.n()) / c.n())
            uct_value = exploit + self.config.c_param * explore
            choices_weights.append(uct_value)

        idx = np.argmax(choices_weights)
        return valid_children[idx]

    def backpropagate(self, value):
        self._number_of_visits += 1
        self._results.append(value)
        if self.parent:
            self.parent.backpropagate(value)

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node(): 
            current_node.update_n_total_children(self.config.width_increase_factor)  
            if not current_node.is_fully_expanded():  
                return current_node.expand(), True  # Expansion
            else:
                current_node = current_node.best_child()  
                if current_node is None:
                    return self, False
        return current_node, False

    def add_value(self, is_additional=False):
        # Return only rv and corresponding reward (same as rv)
        rv, raw_value = self.state.get_value()
        return rv, raw_value

    def add_simulate(self):
        rv, value, smi, sent = self.fast_rollout_evaluation()
        # Cache "best Simulation sample" (only if valid and better)
        try:
            if rv is not None and float(rv) > 0:
                if self._best_rollout_rv is None or float(rv) > float(self._best_rollout_rv):
                    self._best_rollout_rv = float(rv)
                    self._best_rollout_smi = smi
                    self._best_rollout_sentence = sent
        except Exception:
            pass
        return rv, value

    def fast_rollout_evaluation(self):
        action, smiles_answer, has_end_token = self.state.generate_fragment(
            cur_molecule=self.state.cur_molecule,
            is_simulation=True
        )
        _, smiles = sentence2mol(smiles_answer, True)
        rv, value = self.state.get_reward(smiles)
        # Return (rv, value, smi, cur_sentence) for external caching
        return rv, value, smiles, smiles_answer

    def update_n_total_children(self, increase_factor):
        if not self.children:
            return
        values = [np.sum(child.q()) / child.n() for child in self.children]
        values = np.array(values)
        mean_value = np.mean(values)
        diff_values = np.abs(values - mean_value)
        value_diff = np.max(diff_values)
        if value_diff > self.max_q_diff:
            self.max_q_diff = value_diff

        new_n_total_children = min(int(increase_factor * value_diff), 10)
        if new_n_total_children > self.n_total_children_adaptive:
            self.n_total_children_adaptive = new_n_total_children

    def best_action_global_leaf(self):
        """Return global best leaf.
        Convention: "non-terminal" nodes with no children are also considered leaves to avoid returning None.
        """
        # If terminal or no children, consider as leaf and return self
        if self.is_terminal_node() or not self.children:
            return self

        best_leaf = None
        highest_reward = float('-inf')

        for child in self.children:
            leaf = child.best_action_global_leaf()
            if leaf is None:
                continue
            current_reward = max(leaf.result()) if leaf.result() else 0
            if current_reward > highest_reward:
                highest_reward = current_reward
                best_leaf = leaf

        # Fallback: if not found (extreme case), return self
        return best_leaf if best_leaf is not None else self

    def best_child_greedy(self):
        if not self.children:
            return None
        choices = [c.q() / c.n() if c.n() > 0 else 0 for c in self.children]
        idx = np.argmax(choices)
        return self.children[idx]

    def best_action_greedy(self):
        leaf = self.best_action_greedy_leaf()
        rv = leaf._values if leaf._values is not None else 0.0
        _, smi = sentence2mol(leaf.state.cur_sentence, True)

        return rv, smi, leaf.state.cur_sentence

    def best_action_greedy_leaf(self):
        current_node = self
        while not current_node.is_terminal_node():
            next_node = current_node.best_child_greedy()
            if next_node is None:
                break
            current_node = next_node
        return current_node

    def get_end_state(self):
        end_state = self.state.take_action_end(is_greedy=False)
        return end_state

    def generate_all_paths(self):
        all_paths = []
        all_path_set = set()
        queue = deque(self.children)
        while queue:
            cur = queue.popleft()
            cur_path = cur.state.cur_molecule
            if cur_path in all_path_set:
                continue
            all_paths.append({
                "path": cur_path,
                "depth": cur.depth,
                "score": cur.get_cached_reward(),
                "is_terminal": cur.is_terminal_node()
            })
            all_path_set.add(cur_path)
            queue.extend(cur.children)
        return all_paths



class MCTS:
    def __init__(self, initial_state, config, args=None):
        self.initial_state = initial_state

        self.config = config
        self.args = args

        self.root = None
        self.max_search_depth = 0
        self.unique_nodes = set()
        self.time_taken = 0

        # Trace recording: store only lightweight info related to search process (no model weights)
        self._trace = []  # Event dictionary for each step
        self._trace_meta = {}

    def _log_event(self, ev: dict):
        """Append a search event (basic types only, avoid weights/large objects)."""
        try:
            # Ensure serialization friendly
            self._trace.append(dict(ev))
        except Exception:
            pass

    def get_trace(self):
        """Return full search trace (event list)."""
        return list(self._trace)

    def get_trace_meta(self):
        return dict(self._trace_meta)

    def run_mcts(self):
        if self.root is None:
            self.root = MonteCarloTreeSearchNode(state=self.initial_state,
                                                 config=self.config,
                                                 depth=0,
                                                 node_id='root')

        search_iter = 0
        n_terminals = 0

        n_steps, n_rollouts, n_requests = 0, 0, 0

        pbar = tqdm(range(self.config.search_time), desc="MCTS simulations", leave=True)

        # Record meta info
        self._trace_meta = {
            "search_time": int(getattr(self.config, 'search_time', 0)),
            "min_terminals": int(getattr(self.config, 'min_terminals', -1)),
            "max_split_depth": int(getattr(self.config, 'max_split_depth', -1)),
            "init_children": int(getattr(self.config, 'init_children', -1)),
            "n_total_children": int(getattr(self.config, 'n_total_children', -1)),
            "c_param": float(getattr(self.config, 'c_param', 0.0)),
            "n_simulations": int(getattr(self.config, 'n_simulations', 0)),
            "fastrollout_weight": float(getattr(self.config, 'fastrollout_weight', 0.0)),
            "diversity_threshold": float(getattr(self.config, 'diversity_threshold', 0.0)),
            "max_resample_on_empty": int(getattr(self.config, 'max_resample_on_empty', 0)),
            "ts_start": time.time(),
        }

        while search_iter < self.config.search_time or n_terminals < self.config.min_terminals:
            search_iter += 1
            pbar.update(1)
            v, is_expand = self.root._tree_policy()  # Selection & Expansion

            # Record selection path (sequence of child indices from root to current node)
            try:
                sel_path = v.get_ancestor_child_indices()
            except Exception:
                sel_path = []
            self._log_event({
                "iter": int(search_iter),
                "type": "select",
                "node_id": getattr(v, 'node_id', None),
                "depth": int(getattr(v, 'depth', -1) or -1),
                "path": sel_path,
            })

            if is_expand:
                reward = 0.0

                if self.config.value_weight > 0:
                    rv, raw_value = v.add_value(is_additional=False)
                    reward += self.config.value_weight * raw_value

                if self.config.n_simulations > 0 and self.config.fastrollout_weight > 0:
                    if v.is_terminal_node():  # Terminal node: evaluate value directly
                        rv, raw_value = v.add_value(is_additional=False)
                        reward += self.config.fastrollout_weight * raw_value
                    else:
                        # Multiple completions and aggregation: use top (max)
                        rollout_vals = []
                        rollout_rvs = []
                        n_sims = int(self.config.n_simulations)
                        for _ in range(max(1, n_sims)):
                            rvi, vali = v.add_simulate()
                            rollout_rvs.append(float(rvi))
                            rollout_vals.append(float(vali))
                        if len(rollout_vals) == 0:
                            rv = 0.0
                            raw_value = 0.0
                        else:
                            # Take best value from single run simulation (top-1)
                            top_idx = int(np.argmax(rollout_vals))
                            raw_value = float(rollout_vals[top_idx])
                            rv = float(rollout_rvs[top_idx])
                        reward += self.config.fastrollout_weight * raw_value

                v.set_cached_reward(rv, reward)
                v.backpropagate(reward)

                # Record expansion details (lightweight fields only)
                self._log_event({
                    "iter": int(search_iter),
                    "type": "expand",
                    "node_id": getattr(v, 'node_id', None),
                    "depth": int(getattr(v, 'depth', -1) or -1),
                    "rv": None if rv is None else float(rv),
                    "reward": float(reward) if reward is not None else None,
                    "n_visits": int(v.n()),
                    "q_sum": float(v.q()) if v.q() is not None else 0.0,
                    "n_children": int(len(v.children)),
                    "sentence": v.state.cur_sentence,
                    "terminal": bool(v.is_terminal_node()),
                })

                parent_action = v.parent_action if v.parent_action else ""
  
                n_action_steps = 1
                n_steps += n_action_steps
                n_rollouts += int(self.config.n_simulations if (self.config.n_simulations > 0 and not v.is_terminal_node()) else 0)
                n_requests += v.n_repeat_by_parent * 1

                if v.is_terminal_node():
                    n_terminals += 1

            else: # No expansion (e.g., tree too deep or reached unexpandable terminal)
                reward = v.get_cached_reward() # Backpropagate cached reward
                v.backpropagate(reward)  # Backpropagation
                self._log_event({
                    "iter": int(search_iter),
                    "type": "backprop_cached",
                    "node_id": getattr(v, 'node_id', None),
                    "depth": int(getattr(v, 'depth', -1) or -1),
                    "reward": float(reward) if reward is not None else None,
                })

            if v.depth > self.max_search_depth:
                self.max_search_depth = v.depth

        best_leaf = self.root.best_action_global_leaf()
        if best_leaf is not None:
            if getattr(best_leaf, "_best_rollout_sentence", None) is not None and getattr(best_leaf, "_best_rollout_smi", None) is not None:
                final_sentence = best_leaf._best_rollout_sentence
                final_smi = best_leaf._best_rollout_smi
                if getattr(best_leaf, "_best_rollout_rv", None) is not None:
                    final_rv = float(best_leaf._best_rollout_rv)
                else:
                    final_rv = None if best_leaf._values is None else float(best_leaf._values)
            else:
                final_sentence = best_leaf.state.cur_sentence
                _, final_smi = sentence2mol(final_sentence, True)
                final_rv = None if best_leaf._values is None else float(best_leaf._values)
        else:
            final_sentence = ""
            final_smi = None
            final_rv = None
        pbar.close()

        self.total_rollouts = n_rollouts
        self.total_steps = n_steps
        self.total_requests = n_requests

        self._log_event({
            "type": "final",
            "rv": None if final_rv is None else float(final_rv),
            "smi": final_smi,
            "sentence": final_sentence,
            "total_steps": int(self.total_steps),
            "total_rollouts": int(self.total_rollouts),
            "total_requests": int(self.total_requests),
        })

        return final_rv, final_smi, final_sentence

    def run(self):
        start_time = time.time()
        rv, smi, cur_sentence = self.run_mcts()
        end_time = time.time()
        self.time_taken = end_time - start_time
        print(f"run_time:{self.time_taken / 60 :.2f}min")
        return rv, smi, cur_sentence

    def get_time(self):
        return self.time_taken

    def get_max_search_depth(self):
        return self.max_search_depth

    def get_all_paths(self):
        return self.root.generate_all_paths() if self.root else []

    def get_final_state_greedy(self):
        if not self.root:
            return None
        greedy_leaf = self.root.best_action_greedy()
        return greedy_leaf.get_end_state()

    def get_final_state_global(self):
        if not self.root:
            return None
        best_leaf = self.root.best_action_global_leaf()
        return best_leaf.get_end_state()

    def save_tree(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.root, f)

    @classmethod
    def load_tree(cls, filename, config):
        with open(filename, 'rb') as f:
            root = pickle.load(f)
        mcts_recover = cls(initial_state=None, config=config)
        mcts_recover.root = root
        return mcts_recover
