import re
from utils.chem_utils import get_morgan_fingerprint, is_too_similar_to_children, sentence2mol, get_sa, get_qed
import time
import numpy as np
from collections import deque
from rdkit import Chem
import torch
from tqdm import tqdm
import pickle

class MCTSConfig:

    # optimization parameters
    value_weight = 0    # weight of value in the total reward. 0 means no value.
    search_time = 500    # total search times (equal or larger than than the number of nodes expanded)
    min_terminals = -1    # minimum number of terminals must search
    max_split_depth = 10    # maximum depth to split the tree. If larger, only single path will be expanded. If -1, no limit. This is a piror knowledge of the problem.
    init_children = 20     # initial number of children to expand at the root node. if -1, use N_TOTAL_CHILDREN. This is a piror knowledge of the problem.
    n_total_children = 8    # number of children to expand at each node
    c_param = 5    # exploration parameter
    width_increase_factor = 2   # increase the width of the tree by this factor in Adaptive child allocation

    add_value_weight = 0.0
    n_simulations = 1
    fastrollout_weight = 1.0

    greedy_path = False
    max_n_repeat = 5

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MolecularProblemState:

    def __init__(self,
                 model,
                 tokenizer,
                 predictor,
                 cur_molecule=None, 
                 cur_step=0, 
                 max_steps=10,  
                 is_terminate=False,  
                 rewards=None,  
                 has_optimized=False):  

        self.predictor = predictor
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
        _, smiles = sentence2mol(self.cur_sentence, True)
        (rv, rq, rs), value = self.get_reward(smiles)
        return (rv, rq, rs), value
    
    def get_reward(self, smiles):
        if smiles is None:
            return (-1.0, -1.0, -1.0), -1.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (-1.0, -1.0, -1.0), -1.0

        rq = get_qed(mol)
        rs = get_sa(mol)
        rv = 0

        if self.predictor.protein == 'parp1':
            hit_thr = 10.
        elif self.predictor.protein == 'fa7':
            hit_thr = 8.5
        elif self.predictor.protein == '5ht1b':
            hit_thr = 8.7845
        elif self.predictor.protein == 'braf':
            hit_thr = 10.3
        elif self.predictor.protein == 'jak2':
            hit_thr = 9.1
        elif self.predictor.protein == '8g46_protein':
            hit_thr = 2.0

        reward = 0


        rq_ok = rq > 0.5
        rs_ok = rs > (10.0 - 5) / 9  

        if rq_ok:
            reward += 1
        if rs_ok:
            reward += 1


        if rq_ok and rs_ok:
            result = self.predictor.predict([smiles])
            rv = result[0]
            if rv > 0:  
                return (-1.0, -1.0, -1.0), -1.0
            rv = -rv 
            if rv > hit_thr:
                excess = rv - hit_thr
                reward += 2 + excess

        return (rv, rq, rs), reward

    def cond_actions(self, to_end=False, is_greedy=False):
        n_attempts = 5
        for attempt in range(n_attempts):
            try:
                if to_end:
                    action, smiles_answer, has_end_token = self.action2end(is_greedy=is_greedy)  
                else:
                    action, smiles_answer, has_end_token = self.actions(is_greedy=is_greedy)
                    if len(action) == 0:
                        continue
                return action, smiles_answer, has_end_token
            except Exception as e:
                if attempt < n_attempts - 1:
                    print(f'Retry {attempt}, error: {type(e).__name__}', flush=True)
                    continue
                else:
                    raise e

    def actions(self, is_greedy=False):
        temperature = 0.0 if is_greedy else 1.0
        action, smiles_answer, has_end_token = self.generate_fragment(
            cur_molecule=self.cur_molecule,
            max_seq_len=1024,
            temperature=temperature,
            top_k=None,
            stream=False,
            rp=1.0,
            kv_cache=True,
            is_simulation=False
        )
        return action, smiles_answer, has_end_token

    def take_action(self, action):
        new_answer = torch.as_tensor(action, dtype=self.cur_molecule.dtype, device=self.cur_molecule.device).unsqueeze(0)
        next_state = MolecularProblemState(
            model=self.model,
            tokenizer=self.tokenizer,
            predictor=self.predictor,
            cur_molecule=new_answer,
            cur_step=self.cur_step + 1,
            max_steps=self.max_steps,
            is_terminate=False  
        )
        return next_state

    def action2end(self, is_greedy):
        temperature = 0.0 if is_greedy else 1.0
        action, smiles_answer, has_end_token = self.generate_fragment(
            cur_molecule=self.cur_molecule,
            max_seq_len=1024,
            temperature=temperature,
            top_k=None,
            stream=False,
            rp=1.0,
            kv_cache=True,
            is_simulation=True
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
            cur_molecule=answer_updated,
            cur_step=self.cur_step + n_steps,
            max_steps=1000, 
            is_terminate=True
        )
        return end_state

    def generate_fragment(self, cur_molecule, max_seq_len, temperature, top_k, stream, rp, kv_cache, is_simulation):
        with torch.no_grad():
            res_y = self.model.generate(cur_molecule, self.tokenizer, max_new_tokens=max_seq_len,
                                        temperature=temperature, top_k=top_k, stream=stream, rp=rp, kv_cache=kv_cache,
                                        is_simulation=is_simulation)
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")

            history_idx = 0
            complete_answer = cur_molecule[0].tolist()  

            while y != None:
                answer = y[0].tolist()
                complete_answer += answer[history_idx:]

                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

        smiles_answer = self.tokenizer.decode(complete_answer)
        has_end_token = False
        if "[EOS]" in smiles_answer:
            has_end_token = True

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

        self._values = []  
        self._cached_reward = 0.  

   
        self.depth = depth
        self.node_id = node_id
        self.n_repeat_by_parent = n_repeat_by_parent
        self.n_repeat = 0
    
        if self.config.max_split_depth < 0:

            self.config.max_split_depth = self.depth
        if self.depth == 0:
            self.n_total_children_adaptive = self.config.init_children if self.config.init_children > -1 else self.config.init_children
        elif self.depth > self.config.max_split_depth:
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
        return self._values[0] if len(self._values) > 0 else None

    def set_cached_reward(self, rv, rq, rs, raw_value):
        self._values = (rv, rq, rs)
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
        children_fps = []
        for child in self.children:
            child_mol, child_smiles = sentence2mol(child.state.cur_sentence, True)
            fp = get_morgan_fingerprint(child_mol)
            if fp is not None:
                children_fps.append(fp)
        n_repeat = 0

        to_end = self.config.max_split_depth <= (self.depth + 1)
 
        is_greedy = self.config.greedy_path and len(self.children) == 0

        while True:
            action, smiles_answer, has_end_token = self.state.cond_actions(
                to_end=to_end,
                is_greedy=is_greedy,
            )

            new_mol, _ = sentence2mol(smiles_answer, True)

            new_fp = get_morgan_fingerprint(new_mol)
            if new_fp is None:
                n_repeat += 1
                if n_repeat >= self.config.max_n_repeat:
                    break
                continue

            if not is_too_similar_to_children(new_fp, children_fps, threshold=0.8):
                break
            else:
                n_repeat += 1
                if n_repeat >= self.config.max_n_repeat:
                    break

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
                return current_node.expand(), True
            else:
                current_node = current_node.best_child()  
                if current_node is None:
                    return self, False
        return current_node, False

    def add_value(self, is_additional=False):
        (rv, rq, rs), raw_value = self.state.get_value()
        return (rv, rq, rs), raw_value

    def add_simulate(self):
        (rv, rq, rs), value = self.fast_rollout_evaluation()

        return (rv, rq, rs), value

    def fast_rollout_evaluation(self):
        action, smiles_answer, has_end_token = self.state.generate_fragment(
            cur_molecule=self.state.cur_molecule,
            max_seq_len=1024,
            temperature=1.0,
            top_k=None,
            stream=False,
            rp=1.0,
            kv_cache=True,
            is_simulation=True
        )
        _, smiles = sentence2mol(smiles_answer, True)
        (rv, rq, rs), value = self.state.get_reward(smiles)

        return (rv, rq, rs), value

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
        if self.is_terminal_node():
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

        return best_leaf

    def best_child_greedy(self):
        if not self.children:
            return None
        choices = [c.q() / c.n() if c.n() > 0 else 0 for c in self.children]
        idx = np.argmax(choices)
        return self.children[idx]

    def best_action_greedy(self):
        leaf = self.best_action_greedy_leaf()
        rv, rq, rs = leaf._values
        _, smi = sentence2mol(leaf.state.cur_sentence, True)

        return rv, rq, rs, smi, leaf.state.cur_sentence

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

    def run_mcts(self):
        if self.root is None:
            self.root = MonteCarloTreeSearchNode(state=self.initial_state,
                                                 config=self.config,
                                                 depth=0,
                                                 node_id='root')

        search_iter = 0
        n_terminals = 0

        n_steps, n_rollouts, n_requests = 0, 0, 0

        pbar = tqdm(range(self.config.search_time),
                    desc="MCTS simulations",
                    leave=True)

        while search_iter < self.config.search_time or n_terminals < self.config.min_terminals:
            search_iter += 1
            pbar.update(1)
            v, is_expand = self.root._tree_policy()

            if is_expand:
                reward = 0.0

                if self.config.value_weight > 0:
                    (rv, rq, rs), raw_value = v.add_value(is_additional=False)
                    reward += self.config.value_weight * raw_value

                if self.config.n_simulations > 0 and self.config.fastrollout_weight > 0:
                    if v.is_terminal_node():
                        (rv, rq, rs), raw_value = v.add_value(is_additional=False)
                        reward += self.config.fastrollout_weight * raw_value
                    else:
                        (rv, rq, rs), raw_value = v.add_simulate()
                        reward += self.config.fastrollout_weight * raw_value

                v.set_cached_reward(rv, rq, rs, reward)
                v.backpropagate(reward)

                parent_action = v.parent_action if v.parent_action else ""
  
                n_action_steps = parent_action.count(13) - 1   
                n_steps += n_action_steps
                n_rollouts += 1
                n_requests += v.n_repeat_by_parent * n_action_steps

                if v.is_terminal_node():
                    n_terminals += 1

            else:
                reward = v.get_cached_reward()
                v.backpropagate(reward)

            if v.depth > self.max_search_depth:
                self.max_search_depth = v.depth

        best_leaf = self.root.best_action_global_leaf()
        rv, rq, rs = best_leaf._values
        _, smi, = sentence2mol(best_leaf.state.cur_sentence, True)
        cur_sentence = best_leaf.state.cur_sentence
        pbar.close()

        self.total_rollouts = n_rollouts
        self.total_steps = n_steps
        self.total_requests = n_requests

        return rv, rq, rs, smi, cur_sentence

    def run(self):
        start_time = time.time()
        rv, rq, rs, smi, cur_sentence = self.run_mcts()
        end_time = time.time()
        self.time_taken = end_time - start_time
        print(f"run_time:{self.time_taken / 60 :.2f}min")
        return rv, rq, rs, smi, cur_sentence

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




