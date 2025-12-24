from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Union, Iterable
import random
import pickle

DEFAULT_NAME = "Oratrice Mecanique"

class QLearningAI(ABC):
    def __init__(self, name: str = DEFAULT_NAME, random_behaviour: bool = False, 
                 pretrained_path: Optional[str] = None):
        self.memory: Dict[Tuple[str, tuple], float] = {}
        if pretrained_path:
            with open(pretrained_path, "rb") as f:
                self.memory = pickle.load(f)

        self.name = name
        self.random = random_behaviour
        self.cache: Dict[tuple, tuple] = {}
        self.CACHESIZE = self.get_cache_size()

    def select_move(self, board: Union[List, Tuple], token: str, epsilon: float) -> tuple:
        if self.random or random.uniform(0, 1) < epsilon:
            return random.choice(self.get_moves(board))

        all_moves = [(m, self.state_action_value(board, m)) for m in self.get_moves(board)]
        
        # X maximizes Q, O minimizes Q (Zero-sum)
        if token == "X":
            best_move = max(all_moves, key=lambda x: x[1])
        else:
            best_move = min(all_moves, key=lambda x: x[1])

        if best_move[1] == 0.0:
            return random.choice([m[0] for m in all_moves])
        return best_move[0]

    def state_action_value(self, board: tuple, action: tuple) -> float:
        canonical_key, canonical_move, _ = self.get_canonical_key_move_nboard(board, action, board)
        return self.memory.get((canonical_key, canonical_move), 0.0)

    def update_memory(self, s: tuple, a: tuple, s_: tuple, r: float, 
                      alpha: float, gamma: float, token: str):
        key, move, next_board = self.get_canonical_key_move_nboard(s, a, s_)
        old_q = self.state_action_value(s, a)
        future_q = self.future_estimate(next_board, token)
        
        td_target = r + gamma * future_q
        self.memory[(key, move)] = old_q * (1 - alpha) + alpha * td_target

    def get_canonical_key_move_nboard(self, c_board: tuple, move: tuple, n_board: tuple) -> Tuple[str, tuple, tuple]:
        state_triplet = (c_board, move, n_board)
        if state_triplet in self.cache:
            return self.cache[state_triplet]

        forms = self.get_canonical_forms(c_board, move, n_board)
        if not forms:
            return self.memory_key(c_board), move, n_board

        for nb, nm, nnb in forms:
            key = self.memory_key(nb)
            if (key, nm) in self.memory:
                self._update_cache(state_triplet, (key, nm, nnb))
                return key, nm, nnb

        key = self.memory_key(c_board)
        self._update_cache(state_triplet, (key, move, n_board))
        return key, move, n_board

    def _update_cache(self, triplet: tuple, result: tuple):
        if len(self.cache) >= self.CACHESIZE:
            self.cache.clear()
        self.cache[triplet] = result

    def future_estimate(self, board: tuple, token: str) -> float:
        moves = self.get_moves(board)
        if not moves: return 0.0
        # Estimate next state from perspective of opponent
        vals = [self.state_action_value(board, m) for m in moves]
        return min(vals) if token == "X" else max(vals)

    def save_model(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.memory, f)

    @abstractmethod
    def get_moves(self, board: tuple) -> List[tuple]: pass
    @abstractmethod
    def memory_key(self, board: tuple) -> str: pass
    @abstractmethod
    def get_reward_for_move(self, *args) -> float: pass
    @abstractmethod
    def get_canonical_forms(self, *args) -> Iterable: pass
    @abstractmethod
    def get_cache_size(self) -> int: pass
