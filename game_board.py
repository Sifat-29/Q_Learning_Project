from .Minimax import MinimaxTictactoe
from .Q_Learning_AI import QLearningAI
# from Q_Learning_AI_Tictactoe import TictactoeQLearning
# from Q_Learning_AI_Connect4 import Connect4QLearning
from abc import ABC,abstractmethod
import random
import copy
#=======================================================================================================================
#=================================== Game Board Class ==================================================================
#=======================================================================================================================
class GameBoard(ABC):
    """
    Player1 always moves first
    """
    def __init__(self, player1 = "human1", player2 = "human2"):

        # Handling both type of the players
        if isinstance(player1, (str, QLearningAI, MinimaxTictactoe)):
            self.player1 = player1
        else:
            print("Invalid player1, Assigning Human")
            self.player1 = "human1"

        if isinstance(player2, (str, QLearningAI, MinimaxTictactoe)):
            self.player2 = player2
        else:
            print("Invalid player2, Assigning Human")
            self.player2 = "human2"

        # Keeping track of the game winner ("won": player(object), "draw": 0, "not over": None) and moves played
        self.winner = None
        self.moves_played = 0

        # Board depending upon the game being played
        self.board = self.get_empty_board()


    @abstractmethod
    def get_empty_board(self):
        pass

    @abstractmethod
    def get_pretty_printing_board(self, board):
        pass


    def start_game(self, mode="play", test=False, epsilon=0.1, alpha=0.2, gamma=0.9):
        """
        Plays the game based on the type of players (Independent of the game type)
        """
        if not test: print(f"================= Game of {self.game} started between {self.player1} & {self.player2} =================\n")

        if mode == "play": # Simply plays the game instead of training
            while self.winner is None:
                player, other_player, token, token_ = self.get_round_info()

                if not test:
                    print(f"Current Board:")
                    print(self.get_pretty_printing_board(self.board))
                    print(f"\n\nCurrent player is {player}")

                if isinstance(player, str):
                    move = self.make_move(self.board, player, token)
                else:
                    move = self.make_move(tuple((tuple(x) for x in self.board)), player, token)
                self.place_token_on_board(self.board, token, move)
                self.moves_played += 1
                game_status, winning_token = self.is_game_over(self.board, move, check="special")

                if game_status == "won":
                    if not test:
                        print(self.get_pretty_printing_board(self.board))
                        print(f"{player} Won Against {other_player}!!!")
                    self.winner = player

                elif game_status == "draw":
                    if not test:
                        print(self.get_pretty_printing_board(self.board))
                        print(f"{player} Drew Against {other_player}")
                    self.winner = 0

                if not test: print("==============================================================================================")

        elif mode == "train":
            if isinstance(self.player1, str) or isinstance(self.player2, str):
                print("Only AI are allowed to train")
                return

            while self.winner is None:
                player, other_player, token, token_ = self.get_round_info()
                current_board = tuple((tuple(row) for row in self.board))

                # Board after move
                move = self.make_move(current_board, player, token, epsilon=epsilon)
                self.place_token_on_board(self.board, token, move)
                next_board = tuple((tuple(row) for row in self.board))

                self.moves_played += 1

                status, winning_token = self.is_game_over(next_board, move, check="board")

                if isinstance(player, QLearningAI):
                    reward = player.get_reward_for_move(status, winning_token, token, current_board, move, next_board)
                    reward = -reward if token == "O" else reward

                if status == "won" and winning_token == token:
                    self.winner = player
                elif status == "won" and winning_token != token:
                    self.winner = other_player
                elif status == "draw":
                    self.winner = 0

                if isinstance(player, QLearningAI):
                    if not player.is_ai_random():
                        player.update_memory(current_board, tuple(move), next_board, reward, alpha, gamma, token)


    def get_round_info(self):
        """
        Returns current_player, other_player, current_token, other_token
        """
        if self.moves_played % 2 == 0:
            return self.player1, self.player2, "X", "O"
        else:
            return self.player2, self.player1, "O", "X"


    def make_move(self, board, player, token, epsilon=0):
        """
        Make the move for each game board after receiving the input move (move is different for each game)
        TicTacToe: move is (i, j) list/tuple
        Connect4: move is an (i) list/tuple
        Othello: move is an (i, j) list/tuple
        Chess: Haven't figured out yet
        TODO: Othello (P1), Chess (P2)
        """
        if isinstance(player, str):
            return self.get_move_from_player()

        elif isinstance(player, QLearningAI):
            return list(player.select_move(board, token, epsilon))

        elif isinstance(player, MinimaxTictactoe):
            return player.get_optimal_move(board, token)

        else:
            print("Invalid player")
            return None


    @abstractmethod
    def get_move_from_player(self):
        """
        Get a valid move from player depending upon the game type
        TicTacToe: move is (i, j) list
        Connect4: move is a (i) list
        Othello: move is an (i, j) list
        Chess: Haven't figured out yet
        TODO: Chess (P2)
        """
        pass

    @abstractmethod
    def place_token_on_board(self, board, token, move):
        pass

    @abstractmethod
    def is_game_over(self, board, move,check="special"):
        """
        Check whether the game is over
        returns "won" if the game is over and won
        returns "draw" if the game is over and draw
        returns "not over" if the game is not over
        TODO othello(P1), chess(P2)
        """
        pass


