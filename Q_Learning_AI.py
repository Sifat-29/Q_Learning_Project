from abc import ABC,abstractmethod
import random
import pickle

DEFAULT_NAME = "Oratrice Mecanique d'Analyse Cardinale"


class QLearningAI(ABC):
    """
    AI which will be trained through Q-Learning
    """
    def __init__(self, name=DEFAULT_NAME, random_behaviour=False, pretrained_q_memory_path=None):
        if pretrained_q_memory_path is None:
            self.memory = dict()
        else:
            if not isinstance(pretrained_q_memory_path, str):
                print("Give string as path, taking empty memory")
                self.memory = dict()
            else:
                with open(rf"{pretrained_q_memory_path}", "rb") as f:
                    self.memory = pickle.load(f)

        self.name = name
        self.random = random_behaviour
        self.random_moves = 0
        self.nonrandom_moves = 0
        self.cache = dict()
        self.CACHESIZE = self.get_cache_size()
        self.cache_hit = 0
        self.cache_miss = 0


    def is_ai_random(self):
        return self.random

    def select_move(self, board, token, epsilon):
        """
        ε‑greedy + true random tie‑break.

        - With probability ε, pick a random available move.
        - Otherwise, compute Q for each move and pick uniformly among the best (or worst) Q-value.
        """
        if self.random or random.uniform(0,1) < epsilon:
            self.random_moves += 1
            return random.choice(self.get_moves(board))

        all_moves = [(move, self.state_action_value(board, move)) for move in self.get_moves(board)]
        if token == "X":
            m = max(all_moves, key=lambda x: x[1])
        else:
            m = min(all_moves, key=lambda x: x[1])
        self.nonrandom_moves += 1

        if m[1] == 0.0:
            return random.choice(all_moves)[0]
        return m[0]



    def state_action_value(self, board: list|tuple, action: tuple):
        """
        Return the Q-value for (board, action).
        board is a 2d list
        Internally canonicalizes to the folded string key.
        """
        canonical_key, canonical_move, _ = self.get_canonical_key_move_nboard(board, action, board)

        return self.memory.get((canonical_key, canonical_move), 0)


    def update_memory(self, s, a, s_, r, alpha, gamma, token):
        key, move, next_board = self.get_canonical_key_move_nboard(s, a, s_)

        old_q = self.state_action_value(s, a)
        future_q = self.future_estimate(next_board, token)
        td_target = r + gamma * future_q

        self.memory[(key, move)] = old_q * (1 - alpha) + alpha * td_target


    def get_canonical_key_move_nboard(self, c_board, move, n_board):
        if (c_board, move ,n_board) in self.cache:
            self.cache_hit += 1
            return self.cache[(c_board, move, n_board)]

        forms = self.get_canonical_forms(c_board, move, n_board)

        if not forms:
            return self.memory_key(c_board), move, n_board

        for new_board, new_move, new_n_board in forms:
            key = self.memory_key(new_board)
            if (key, new_move) in self.memory:
                self.add_to_cache(c_board, move, n_board, key, new_move, new_n_board)
                return key, new_move, new_n_board

        key = self.memory_key(c_board)
        self.add_to_cache(c_board, move, n_board, key, move, n_board)
        return key, move, n_board


    def add_to_cache(self, prev_board, move, next_board, prev_board_key, new_move, new_next_board):
        self.cache_miss += 1
        if len(self.cache) >= self.CACHESIZE:
            self.cache.clear()

        self.cache[(prev_board, move, next_board)] = (prev_board_key, new_move, new_next_board)


    def future_estimate(self, board, token):
        moves = self.get_moves(board)
        if not moves:
            return 0

        if token == "X":
            return min((self.state_action_value(board, move) for move in moves))
        elif token == "O":
            return max((self.state_action_value(board, move) for move in moves))
        else:
            return 0


    def save_model(self, path=None):
        if path is None or not isinstance(path, str):
            print("Enter a valid path")
            return

        with open(rf"{path}", "wb") as f:
            pickle.dump(self, f)


    @abstractmethod
    def get_moves(self, board):
        """
        Returns a list of all possible moves (different for each game)
        """
        pass

    @abstractmethod
    def memory_key(self, board):
        """
        Takes board which is a 2d list and returns a string to act as memory key
        """
        pass

    @abstractmethod
    def get_reward_for_move(self, status, winning_token, current_token, current_board, move, next_board):
        """
        Returns the reward for the given move
        """
        pass

    @abstractmethod
    def get_canonical_forms(self, c_board, move, n_board):
        pass

    @abstractmethod
    def get_cache_size(self):
        pass

    def __str__(self):
        return f"{self.name} (AI)"

    def __repr__(self):
        return f"{self.name} (AI)"