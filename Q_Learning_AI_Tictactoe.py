from .Q_Learning_AI import QLearningAI, DEFAULT_NAME


class TictactoeQLearning(QLearningAI):
    """
    AI for TicTacToe
    """
    def __init__(self, name=DEFAULT_NAME, random_behaviour = False, rewards=None, pretrained_q_memory=None):
        super().__init__(name, random_behaviour, pretrained_q_memory)
        self.CORNER_MOVES = [(0,0), (2,2), (0,2), (2,0)]
        self.CENTER_MOVE = (1,1)
        self.EDGE_MOVES = [(0,1), (1,0), (1,2), (2,1)]
        self.DIAGONALS = [[(0,0), (1,1), (2,2)], [(0,2), (1,1), (2,0)]]
        self.LINES = [[(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],  # rows
                      [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],  # cols
                      [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)]]
        self.reward_parameters = ["win", "loss", "draw", "move cost", "center occupancy", "corner occupancy", "created 2 in row", "blocked 2 in row", "suicide created", "suicide move"]
        if rewards is None:
            rewards = [1.0, -1.0, -0.16536050827887683, -0.060851765544286104, 0.02, 0.01, 0.06930998784601186, 0.03596896892645857, 0.1, -0.2]

        self.rewards = dict(zip(self.reward_parameters, rewards))


    def get_moves(self, board):
        """
        Returns a list of all possible moves
        """
        moves = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == " ":
                    moves.append((i, j))
        return moves


    def memory_key(self, board):
        l = ["".join(x) for x in board]
        return "".join(l)


    def get_reward_for_move(self, status, winning_token, current_token, current_board, move, next_board):
        """
        Returns the reward for the given move
        """
        if status == "won" and winning_token == current_token:
            return self.rewards["win"] # Winning Reward
        elif status == "won" and winning_token != current_token:
            return self.rewards["loss"] # Losing Reward
        elif status == "draw":
            return self.rewards["draw"]
        else:
            r = self.rewards["move cost"]

            if move == self.CENTER_MOVE:
                r += self.rewards["center occupancy"]
            elif move in self.CORNER_MOVES:
                r += self.rewards["corner occupancy"]

            before_own, before_opp = self.get_board_info(current_board, current_token)
            after_own, after_opp = self.get_board_info(next_board, current_token)

            created = after_own - before_own
            blocked = before_opp - after_opp

            r += created * self.rewards["created 2 in row"]
            r += blocked * self.rewards["blocked 2 in row"]

            # fork bonus
            if created >= 2:
                r += self.rewards["suicide created"]

            r += self.rewards["suicide move"] * max(0, after_opp - before_opp)

            return r


    def get_board_info(self, board, token):
        lines = self.LINES
        opp_token = "X" if token == "O" else "O"
        own_twos = opp_twos = 0

        for coords in lines:
            pieces = [board[r][c] for r, c in coords]
            if pieces.count(token) == 2 and pieces.count(" ") == 1:
                own_twos += 1
            if pieces.count(opp_token) == 2 and pieces.count(" ") == 1:
                opp_twos += 1
        return own_twos, opp_twos



    def get_canonical_forms(self, c_board, move, n_board):
        """
        returns an iterable of all canonical forms
        """
        # Original and vertical mirror
        canonical_forms = [(c_board, move, n_board), self.get_mirror_form(c_board, move, n_board)]

        # Add inverted
        canonical_forms.append(
            self.get_inverted_form(canonical_forms[-1][0], canonical_forms[-1][1], canonical_forms[-1][2]))
        canonical_forms.append(
            self.get_mirror_form(canonical_forms[-1][0], canonical_forms[-1][1], canonical_forms[-1][2]))

        inverted_forms = [self.get_transpose_form(form[0], form[1], form[2]) for form in canonical_forms]

        return canonical_forms + inverted_forms
        # return None

    def get_mirror_form(self, c_board, move, n_board):
        """
        returns a copy of vertical mirror form
        """
        mirror_move = (move[0], 2 - move[1])
        c_board_copy = [list(x) for x in c_board]
        n_board_copy = [list(x) for x in n_board]

        for board in [c_board_copy, n_board_copy]:
            for row in board:
                row[0], row[2] = row[2], row[0]

        return tuple(tuple(x) for x in c_board_copy), mirror_move, tuple(tuple(x) for x in n_board_copy)

    def get_inverted_form(self, c_board, move, n_board):
        """
        returns a copy of inverted form
        """
        mirror_move = (2 - move[0], move[1])
        c_board_copy = [list(x) for x in c_board]
        n_board_copy = [list(x) for x in n_board]

        for board in [c_board_copy, n_board_copy]:
            board[0], board[2] = board[2], board[0]

        return tuple(tuple(x) for x in c_board_copy), mirror_move, tuple(tuple(x) for x in n_board_copy)

    def get_transpose_form(self, c_board, move, n_board):
        new_move = (move[1], move[0])
        c_board_copy = [list(x) for x in c_board]
        n_board_copy = [list(x) for x in n_board]
        for board in [c_board_copy, n_board_copy]:
            board[0][1], board[1][0] = board[1][0], board[0][1]
            board[0][2], board[2][0] = board[2][0], board[0][2]
            board[1][2], board[2][1] = board[2][1], board[1][2]
        return tuple(tuple(x) for x in c_board_copy), new_move, tuple(tuple(x) for x in n_board_copy)

    def get_cache_size(self):
        return 10000