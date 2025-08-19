from common import StateVector 

"""Model Based checkers planner"""

def get_checkers_move(current_state: StateVector, self_color: str):
    """
    Model-based single-move planner for checkers.

    Important to note that this method assumes the opponent is running the same strategy

    Input: current_state: object with attribute `state` (8x8 list of lists of strings)
    Returns: a deepcopy of current_state modified with the chosen move applied.
    """
    import copy

    board = current_state.state  # 8x8 list of lists
    N = 8
    win = False


    # Heuristic evaluator
    def evaluate_board(board):
        # simple material + king weight + advancement
        w_pieces = 0
        b_pieces = 0
        w_kings = 0
        b_kings = 0
        advancement = 0.0  # positive favors white

        for r in range(N):
            for c in range(N):
                p = board[r][c]
                if p == 'w':
                    w_pieces += 1
                    # white advanced towards row 0 -> higher value if closer to promotion
                    advancement += (7 - r) * 0.01
                elif p == 'b':
                    b_pieces += 1
                    advancement -= r * 0.01
                elif p == 'kw':
                    w_kings += 1
                    advancement += (7 - r) * 0.01
                elif p == 'kb':
                    b_kings += 1
                    advancement -= r * 0.01

        material = (w_pieces + 1.5 * w_kings) - (b_pieces + 1.5 * b_kings)
        return material + advancement

    # Opponent color
    def opponent(color):
        return 'b' if color == 'w' else 'w'


    # Main planner:
    color = self_color

    legal_moves = current_state.generate_legal_moves(board, color)
    if not legal_moves:
        # no legal moves => return deep-copy without change
        return copy.deepcopy(current_state)

    best_move = None
    best_score = -1e9

    for move in legal_moves:
        # simulate our move
        next_board = current_state.apply_move_to_board(board, move, color)
        # evaluate after opponent best response (1 ply opponent)
        opp_moves = current_state.generate_legal_moves(next_board, opponent(color)) 
        if not opp_moves:
            # opponent has no moves -> we win: large score
            score = 1e6 + evaluate_board(next_board)
            win = True
        else:
            opp_best = 1e9
            for opp_move in opp_moves:
                opp_board = current_state.apply_move_to_board(next_board, opp_move, opponent(color))
                val = evaluate_board(opp_board)
                if val < opp_best:
                    opp_best = val
            # prefer moves that maximize (our evaluation - opponent_best_evaluation)
            our_val = evaluate_board(next_board)
            score = our_val - opp_best

        if score > best_score:
            best_score = score
            best_move = move

    # apply chosen move to a deep copy of current_state and return
    result_obj = copy.deepcopy(current_state)
    result_obj.state = current_state.apply_move_to_board(board, best_move, color)
    # if your StateVector has a 'to_move' attribute, toggle it
    if hasattr(result_obj, 'to_move'):
        result_obj.to_move = opponent(color)
    return result_obj, win

