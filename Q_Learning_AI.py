from abc import ABC,abstractmethod
import random
import copy
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
        self.unique_counter = 0
        self.nonunique_counter = 0
        self.random = random_behaviour
        self.random_moves = 0
        self.nonrandom_moves = 0
        self.cache = dict()
        self.CACHESIZE = self.get_cache_size()
        self.cache_hit = 0
        self.cache_miss = 0


    def is_ai_random(self):
        return self.random


    def make_move_greedy(self, board):
        """
        returns the best possible move
        """
        if self.random:
            return self.make_move_random(board)

        all_moves = [(move, self.state_action_value(board, move)) for move in self.get_moves(board)]
        m = max(all_moves, key=lambda x: x[1])
        self.nonrandom_moves += 1

        if m[1] == 0:
            # print(f"Unique, {m[1]}")
            self.unique_counter += 1
            return random.choice(all_moves)[0]
        # print(f"NonUnique, {m[1]}")
        self.nonunique_counter += 1
        return m[0]

    def state_action_value(self, board, action):
        """
        Return the Q-value for (board, action).
        Internally canonicalizes to the folded string key.
        """
        canonical_key, canonical_move, _ = self.get_canonical_key_move_nboard(board, action, board)

        # 3) Default to 0.0 if unseen
        return self.memory.get((canonical_key, canonical_move), 0.0)


    def make_move_random(self, board):
        """
        returns a random move from all possible moves
        """
        move = random.choice(self.get_moves(board))

        q = self.state_action_value(board, move)
        if q == 0:
            self.unique_counter += 1
            # print(f"Unique, {q}")
        else:
            self.nonunique_counter += 1
            # print(f"NonUnique, {q}")
        self.random_moves += 1
        return move


    def update_memory(self, s, a, s_, r, alpha, gamma):
        key, move, next_board = self.get_canonical_key_move_nboard(s, a, s_)

        old_q = self.state_action_value(s, a)
        future_q = self.max_state(next_board)
        td_target = r + gamma * future_q

        self.memory[(key, move)] = old_q * (1 - alpha) + alpha * td_target


    def get_canonical_key_move_nboard(self, c_board, move, n_board):
        if (c_board, move ,n_board) in self.cache:
            self.cache_hit += 1
            # self.cache[(c_board, move, n_board)][1] += 1
            # return self.cache[(c_board, move, n_board)][0]
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
        if len(self.cache) > self.CACHESIZE:
            self.cache.clear()
            # sorted_items = sorted(list(self.cache.items()), key=lambda x: x[1][1])
            # retain_count = int(0.5 * len(sorted_items))
            # self.cache = dict(sorted_items[-retain_count:])

        # self.cache[(prev_board, move, next_board)] = [(prev_board_key, new_move, new_next_board), 1]
        self.cache[(prev_board, move, next_board)] = (prev_board_key, new_move, new_next_board)


    def max_state(self, board):
        moves = self.get_moves(board)
        if not moves:
            return 0
        return -max((self.state_action_value(board, move) for move in moves))


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