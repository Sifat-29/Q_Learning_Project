from .game_board import GameBoard

#=======================================================================================================================
#======================================= TicTacToe GameBoard Subclass ==================================================
#=======================================================================================================================
class TictactoeBoard(GameBoard):
    def __init__(self, player1 = "human1", player2 = "human2"):
        self.game = "Tictactoe"
        super().__init__(player1, player2)

    def get_empty_board(self):
        return [[" "] * 3 for _ in range(3)]

    def get_pretty_printing_board(self, board):
        string_list = ["      column\n     0   1   2\nrow\n"]

        for i in range(3):
            string_list.append(f" {i}   {board[i][0]} | {board[i][1]} | {board[i][2]}\n")
            if i != 2:
                string_list.append("    -----------\n")

        return "".join(string_list)

    def get_move_from_player(self):
        while True:
            move = list(map(int, input("Enter value on the grid (0 indexed) to make a move on (zero indexed): ").split()))

            if len(move) != 2 or move[0] < 0 or move[1] < 0 or move[0] > 2 or move[1] > 2 or self.board[move[0]][move[1]] != " ":
                print("Invalid tictactoe move, enter again")
            else:
                return move

    def place_token_on_board(self, board, token, move):
        self.board[move[0]][move[1]] = token

    def is_game_over(self, board, move, check="special"):
        # Row check
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != " ":
                return "won", board[i][0]

        # Column check
        for i in range(3):
            if board[0][i] == board[1][i] == board[2][i] != " ":
                return "won", board[0][i]

        # Diagonal check
        if board[0][0] == board[1][1] == board[2][2] != " ":
            return "won", board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != " ":
            return "won", board[0][2]

        # Checking for draw through moves played
        if check == "special":
            if self.moves_played == 9:
                return "draw", " "
            else:
                return "not over", " "
        else:
            for i in range(len(board)):
                for j in range(len(board[0])):
                    if board[i][j] == " ":
                        return "not over", " "
            return "draw", " "




#=======================================================================================================================
#======================================== Connect4 GameBoard Subclass ==================================================
#=======================================================================================================================
class Connect4Board(GameBoard):
    def __init__(self, player1="human1", player2="human2"):
        self.game = "Connect4"
        super().__init__(player1, player2)

    def get_empty_board(self):
        return [[" "] * 7 for _ in range(6)]

    def get_pretty_printing_board(self, board):
        string_list = ["           column\n  0   1   2   3   4   5   6 \n"]

        for i in range(6):
            string_list.append(f" --- --- --- --- --- --- ---\n")
            ns = "|"
            for j in range(7):
                ns += f" {board[i][j]} |"
            ns += "\n"
            string_list.append(ns)
        string_list.append(" --- --- --- --- --- --- ---\n")

        return "".join(string_list)

    def get_move_from_player(self):
        while True:
            move = list(map(int, input("Enter value on the grid (0 indexed) to make a move on (zero indexed): ").split()))

            if len(move) != 1 or move[0] < 0 or move[0] > 6 or self.board[0][move[0]] != " ":
                print("Invalid connect4 move, enter again")
            else:
                return move

    def place_token_on_board(self, board, token, move):
        column = move[0]

        row = 0
        while row <= 5 and self.board[row][column] == " ":
            row += 1

        self.board[row - 1][column] = token
        move[0]= row - 1
        move.append(column)

    def is_game_over(self, board, move, check="special"):
        row, column = move
        token = board[row][column]

        # Checking for vertical case
        start, end = row - 1, row + 1
        while start >= 0 and board[start][column] == token:
            start -= 1
        while end <= 5 and board[end][column] == token:
            end += 1
        if abs(end - start) >= 5:
            return "won", token

        # Checking for horizontal case
        start, end = column - 1, column + 1
        while start >= 0 and board[row][start] == token:
            start -= 1
        while end <= 6 and board[row][end] == token:
            end += 1
        if abs(end - start) >= 5:
            return "won", token

        # Diagonal check \
        start_row, start_column = row - 1, column - 1
        end_row, end_column = row + 1, column + 1
        while start_row >= 0 and start_column >= 0 and board[start_row][start_column] == token:
            start_row -= 1
            start_column -= 1
        while end_row <= 5 and end_column <= 6 and board[end_row][end_column] == token:
            end_row += 1
            end_column += 1
        if abs(end_row - start_row) >= 5 and abs(end_column - start_column) >= 5:
            return "won", token

        # Diagonal check /
        start_row, start_column = row - 1, column + 1
        end_row, end_column = row + 1, column - 1
        while start_row >= 0 and start_column <= 6 and board[start_row][start_column] == token:
            start_row -= 1
            start_column += 1
        while end_row <= 5 and end_column >= 0 and board[end_row][end_column] == token:
            end_row += 1
            end_column -= 1
        if abs(end_row - start_row) >= 5 and abs(end_column - start_column) >= 5:
            return "won", token

        if check == "special":
            if self.moves_played == 42:
                return "draw", " "
            else:
                return "not over", " "
        else:
            for i in range(6):
                for j in range(7):
                    if board[i][j] == " ":
                        return "not over", " "
            return "draw", " "



#=======================================================================================================================
#======================================== Othello GameBoard Subclass ===================================================
#=======================================================================================================================
class OthelloBoard(GameBoard):
    def __init__(self, player1="human1", player2="human2"):
        self.game = "Othello"
        super().__init__(player1, player2)
        self.piece_info = {"X": 2, "O": 2}

    def get_empty_board(self):
        board = [[" "] * 8 for _ in range(8)]
        board[3][3], board[3][4], board[4][3], board[4][4] = "X", "O", "O", "X"
        return board

    def get_pretty_printing_board(self, board):
        string_list = [
            "                column\n      0   1   2   3   4   5   6   7\nrow  --- --- --- --- --- --- --- ---\n"]

        for i in range(8):
            ns = f" {i}  |"
            for j in range(8):
                ns += f" {board[i][j]} |"
            ns += "\n"
            string_list.append(ns)
            string_list.append("     --- --- --- --- --- --- --- ---\n")

        print("".join(string_list))

    def get_move_from_player(self):
        while True:
            move = list(map(int, input("Enter value on the grid (0 indexed) to make a move on (zero indexed): ").split()))

            if len(move) != 2 or move[0] < 0 or move[1] < 0 or move[0] > 7 or move[1] > 7 or self.board[move[0]][
                move[1]] != " ":
                print("Invalid othello move, enter again")
            else:
                return move

    def place_token_on_board(self, board, token, move):
        """
        Between current placed token and next to straight or diagonal, convert all consecutive others to the placed token
        TODO othello move (P1)
        """
        return

    def is_game_over(self, board, move, check="special"):
        """
        TODO
        """
        return

    def get_canonical_forms(self, c_board, move, n_board):
        """
        returns an iterable of all canonical forms
        """
        pass