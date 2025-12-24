from .Minimax import MinimaxTictactoe
from .Q_Learning_AI import QLearningAI
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Optional, Dict
import random
import copy

class GameBoard(ABC):
    """
    Abstract Base Class for Game Boards.
    Manages the game loop, player turns, and evaluation logic.
    """
    def __init__(self, player1: Union[str, QLearningAI, MinimaxTictactoe] = "human1", 
                 player2: Union[str, QLearningAI, MinimaxTictactoe] = "human2"):

        self.player1 = player1 if isinstance(player1, (str, QLearningAI, MinimaxTictactoe)) else "human1"
        self.player2 = player2 if isinstance(player2, (str, QLearningAI, MinimaxTictactoe)) else "human2"

        self.winner: Optional[Union[str, QLearningAI, MinimaxTictactoe, int]] = None
        self.moves_played: int = 0
        self.board = self.get_empty_board()

    @abstractmethod
    def get_empty_board(self) -> List[List[str]]:
        pass

    @abstractmethod
    def get_pretty_printing_board(self, board: List[List[str]]) -> str:
        pass

    def start_game(self, mode: str = "play", verbose: bool = True, epsilon: float = 0.1, 
                   alpha: float = 0.2, gamma: float = 0.9) -> Dict[str, any]:
        """
        Executes the game loop. 
        Returns a summary of the game outcome.
        """
        if verbose: 
            print(f"--- {self.game} match: {self.player1} vs {self.player2} ---")

        while self.winner is None:
            player, other_player, token, _ = self.get_round_info()

            if mode == "play":
                if verbose:
                    print(self.get_pretty_printing_board(self.board))
                
                state_representation = tuple(tuple(row) for row in self.board) if not isinstance(player, str) else self.board
                move = self.make_move(state_representation, player, token)
                self.place_token_on_board(self.board, token, move)
                self.moves_played += 1
                
                status, winning_token = self.is_game_over(self.board, move, check="special")
                if status == "won":
                    self.winner = player
                elif status == "draw":
                    self.winner = 0

            elif mode == "train":
                current_board = tuple(tuple(row) for row in self.board)
                move = self.make_move(current_board, player, token, epsilon=epsilon)
                self.place_token_on_board(self.board, token, move)
                next_board = tuple(tuple(row) for row in self.board)
                self.moves_played += 1

                status, winning_token = self.is_game_over(next_board, move, check="board")

                if isinstance(player, QLearningAI):
                    reward = player.get_reward_for_move(status, winning_token, token, current_board, move, next_board)
                    # Perspective shift for zero-sum logic
                    reward = -reward if token == "O" else reward
                    if not player.is_ai_random():
                        player.update_memory(current_board, tuple(move), next_board, reward, alpha, gamma, token)

                if status == "won":
                    self.winner = player if winning_token == token else other_player
                elif status == "draw":
                    self.winner = 0

        return {"winner": self.winner, "moves": self.moves_played}

    def get_round_info(self) -> Tuple[any, any, str, str]:
        if self.moves_played % 2 == 0:
            return self.player1, self.player2, "X", "O"
        return self.player2, self.player1, "O", "X"

    def make_move(self, board: Union[List, Tuple], player: any, token: str, epsilon: float = 0) -> List[int]:
        if isinstance(player, str):
            return self.get_move_from_player()
        elif isinstance(player, QLearningAI):
            return list(player.select_move(board, token, epsilon))
        elif isinstance(player, MinimaxTictactoe):
            return player.get_optimal_move(board, token)
        return []

    @abstractmethod
    def get_move_from_player(self) -> List[int]:
        pass

    @abstractmethod
    def place_token_on_board(self, board: List[List[str]], token: str, move: List[int]):
        pass

    @abstractmethod
    def is_game_over(self, board: List[List[str]], move: List[int], check: str = "special") -> Tuple[str, str]:
        pass
