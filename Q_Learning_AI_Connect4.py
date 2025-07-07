from .Q_Learning_AI import QLearningAI, DEFAULT_NAME

class Connect4QLearning(QLearningAI):
    """
    AI for Connect
    """
    def __init__(self, name: str=DEFAULT_NAME, random_behaviour: bool=False, rewards: list=None):
        super().__init__(name, random_behaviour)
        self.segments_4 = {"rows": tuple((tuple(((i, j + k) for k in range(4))) for i in range(6) for j in range(4))),
                           "columns": tuple((tuple(((i + k, j) for k in range(4))) for j in range(7) for i in range(3))),
                           "diagonals": tuple(tuple((i + k, j + k) for k in range(4)) for i in range(3) for j in range(4))
                           + tuple(tuple((i - k, j + k) for k in range(4)) for i in range(3, 6) for j in range(4))}

        if rewards is None:
            rewards = [1, -1, 0, 0.01, 0.1, 0.05, 0.03, 0.2, 0.02, 0.15, 0.25, 0.3, 0.05]
        self.reward_parameters = ["win", "loss", "draw", "move", "center_column", "adj_center_column", "2_one_side_open", "3_one_side_open",
                                  "block_2_one_side_open", "block_3_one_side_open", "create_fork", "opponent_fork", "vertical_stack"]
        self.rewards_dict = dict(zip(self.reward_parameters, rewards))

    def get_moves(self, board):
        """
        Returns a list of all possible moves
        """
        moves = []
        for i in range(len(board[0])):
            if board[0][i] == " ":
                moves.append((i,))

        return moves


    def memory_key(self, board):
        l = ["".join(x) for x in board]
        return "".join(l)


    def get_reward_for_move(self, status, winning_token, current_token, current_board, move, next_board):
        """
        Returns the reward for the given move
        """
        # Change move to be only column in case of c4
        row, col = move
        if row >= 1 and current_board[row][col] == current_token:
            vertical_stack_flag = True
        else:
            vertical_stack_flag = False

        move.pop(0)
        opp_token = "X" if current_token == "O" else "O"

        if status == "won" and winning_token == current_token:
            return self.rewards_dict["win"] # Winning Reward
        elif status == "won" and winning_token != current_token:
            return self.rewards_dict["loss"] # Losing Reward
        elif status == "draw":
            return self.rewards_dict["draw"]
        else:
            r = self.rewards_dict["move"]

            if move[0] == 3: r += self.rewards_dict["center_column"]
            elif move[0] == 4 or move[0] == 2: r += self.rewards_dict["adj_center_column"]

            prev_own_2_open_both_sides, prev_own_3_open_one_sides, prev_opp_2_open_both_sides, prev_opp_3_open_one_sides, prev_own_2_open_one_side, prev_opp_2_open_one_side = self.get_board_info(current_board, current_token, opp_token)
            after_own_2_open_both_sides, after_own_3_open_one_sides, after_opp_2_open_both_sides, after_opp_3_open_one_sides, after_own_2_open_one_side, after_opp_2_open_one_side = self.get_board_info(next_board, current_token, opp_token)

            r += self.rewards_dict["2_one_side_open"] * max(0, after_own_2_open_one_side - prev_own_2_open_one_side)
            r += self.rewards_dict["block_2_one_side_open"] * max(0, after_opp_2_open_one_side - prev_opp_2_open_one_side)
            r += self.rewards_dict["create_fork"] * max(0, after_own_2_open_both_sides - prev_own_2_open_both_sides)
            r += self.rewards_dict["3_one_side_open"] * max(0, after_own_3_open_one_sides - prev_own_3_open_one_sides)
            r += self.rewards_dict["opponent_fork"] * max(0, after_opp_2_open_both_sides - prev_opp_2_open_both_sides)
            r += self.rewards_dict["block_3_one_side_open"] * max(0, after_opp_3_open_one_sides - prev_opp_3_open_one_sides)

            if vertical_stack_flag:
                r += self.rewards_dict["vertical_stack"]

            return r


    def get_board_info(self, board, current_token, opp_token):
        own_2_open_both_sides = opp_2_open_both_sides = 0
        own_3_open_one_sides = opp_3_open_one_sides = 0
        own_2_open_one_sides = opp_2_open_one_sides = 0

        for segments in self.segments_4.values():
            for segment in segments:
                tokens = [board[x[0]][x[1]] for x in segment]

                if tokens.count(current_token) == 2:

                    if tokens[0] == " " and tokens[3] == " " and tokens[1] == current_token and tokens[2] == current_token:
                        own_2_open_both_sides += 1
                    elif tokens.count(" ") == 1:
                        if tokens[0] == current_token and tokens[1] == current_token and tokens[2] == " ":
                            own_2_open_one_sides += 1
                        elif tokens[1] == current_token and tokens[2] == current_token and tokens[0] == " ":
                            own_2_open_one_sides += 1
                        elif tokens[1] == current_token and tokens[2] == current_token and tokens[3] == " ":
                            own_2_open_one_sides += 1
                        elif tokens[1] == current_token and tokens[2] == current_token and tokens[3] == " ":
                            own_2_open_one_sides += 1

                elif tokens.count(opp_token) == 2:

                    if tokens[0] == " " and tokens[3] == " " and tokens[1] == opp_token and tokens[2] == opp_token:
                        opp_2_open_both_sides += 1

                    elif tokens.count(" ") == 1:
                        if tokens[0] == opp_token and tokens[1] == opp_token and tokens[2] == " ":
                            opp_2_open_one_sides += 1
                        elif tokens[1] == opp_token and tokens[2] == opp_token and tokens[0] == " ":
                            opp_2_open_one_sides += 1
                        elif tokens[1] == opp_token and tokens[2] == opp_token and tokens[3] == " ":
                            opp_2_open_one_sides += 1
                        elif tokens[1] == opp_token and tokens[2] == opp_token and tokens[3] == " ":
                            opp_2_open_one_sides += 1

                elif tokens.count(" ") == 1 and tokens.count(current_token) == 3:
                    own_3_open_one_sides += 1
                elif tokens.count(" ") == 1 and tokens.count(opp_token) == 3:
                    opp_3_open_one_sides += 1

        return own_2_open_both_sides, own_3_open_one_sides, opp_2_open_both_sides, opp_3_open_one_sides, own_2_open_one_sides, opp_2_open_one_sides


    def get_canonical_forms(self, c_board, move, n_board):
        """
        TODO
        """
        new_c_board = [list(x) for x in c_board]
        new_n_board = [list(x) for x in n_board]
        new_move = (6 - move[0],)
        for row in new_c_board:
            row.reverse()
        for row in new_n_board:
            row.reverse()
        return (c_board, move, n_board), (tuple(tuple(x) for x in new_c_board), new_move, tuple(tuple(x) for x in new_n_board))


    def get_cache_size(self):
        return 7500000