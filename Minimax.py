import random

class MinimaxTictactoe:
    def __init__(self, suboptimal=False, name="Logos"):
        self.name = name
        self.suboptimal = suboptimal

    def get_optimal_move(self, board, token, epsilon=0.1):
        if self.suboptimal and random.uniform(0, 1) < epsilon:
            return random.choice(self.get_possible_moves(board))

        board_copy = [list(row) for row in board]
        move = self.minimax(board_copy, token, call="move")
        return move

    def minimax(self, board, player_token, call="move", alpha=-float("inf"), beta=float("inf")):
        if self.get_winner(board) == "X":
            return 1
        elif self.get_winner(board) == "O":
            return -1
        elif self.get_winner(board) == "draw":
            return 0

        other_player_token = "X" if player_token == "O" else "O"

        possible_moves = self.get_possible_moves(board)

        best_value_move = None
        if player_token == "X":
            best_value = -float("inf")
            for move in possible_moves:
                result = self.make_move(board, player_token, move)
                value = self.minimax(result, other_player_token, call="value", alpha=alpha, beta=beta)
                if value > best_value:
                    best_value = value
                    best_value_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            best_value = float("inf")
            for move in possible_moves:
                result = self.make_move(board, player_token, move)
                value = self.minimax(result, other_player_token, call="value", alpha=alpha, beta=beta)
                if value < best_value:
                    best_value = value
                    best_value_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        if call == "value":
            return best_value
        else:
            return best_value_move


    def get_winner(self, board):
        # Row check
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != " ":
                return board[i][0]

        # Column check
        for i in range(3):
            if board[0][i] == board[1][i] == board[2][i] != " ":
                return board[0][i]

        # Diagonal check
        if board[0][0] == board[1][1] == board[2][2] != " ":
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != " ":
            return board[0][2]

        # Checking for draw through moves played
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == " ":
                    return "not over"
        return "draw"

    def make_move(self, board, token, move):
        new_board = [row[:] for row in board]
        new_board[move[0]][move[1]] = token
        return new_board

    def get_possible_moves(self, board):
        return [(i, j) for i in range(len(board)) for j in range(len(board[0])) if board[i][j] == " "]

    def __repr__(self):
        if self.suboptimal:
            return f"{self.name} (SUBOPTIMAL AI)"
        else:
            return f"{self.name} (OPTIMAL AI)"
