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

    import copy

    N = 8
    win = False

    # evaluator accepts either a raw board (8x8 list) or a StateVector
    def evaluate_board(board_or_state) -> float:
        board = board_or_state.state if isinstance(board_or_state, StateVector) else board_or_state

        w_pieces = b_pieces = w_kings = b_kings = 0
        advancement = 0.0  # positive favors white

        for r in range(N):
            for c in range(N):
                p = board[r][c]
                if p == 'w':
                    w_pieces += 1
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

    def opponent(color: str) -> str:
        return 'b' if color == 'w' else 'w'

    color = self_color

    # get legal moves for current_state (returns list[Move])
    legal_moves = current_state.get_legal_moves(color)
    if not legal_moves:
        # no legal moves => return deep-copy without change
        return copy.deepcopy(current_state), False

    best_move = None
    best_score = -1e9

    for move in legal_moves:
        # simulate our move -> returns a new StateVector
        next_state: StateVector = current_state.apply_move_to_board(move, color)

        # opponent responses (list[Move]) from the resulting state
        opp_moves = next_state.get_legal_moves(opponent(color))
        if not opp_moves:
            # opponent has no moves -> we win
            score = 1e6 + evaluate_board(next_state)
            local_win = True
        else:
            # opponent will choose move that minimizes our evaluation (worst for us)
            opp_best = 1e9
            for opp_move in opp_moves:
                opp_state = next_state.apply_move_to_board(opp_move, opponent(color))
                val = evaluate_board(opp_state)
                if val < opp_best:
                    opp_best = val
            our_val = evaluate_board(next_state)
            score = our_val - opp_best
            local_win = False

        if score > best_score:
            best_score = score
            best_move = move
            # update global win if this move leads to forced win
            win = local_win

    # apply chosen move to produce final StateVector result
    result_state: StateVector = current_state.apply_move_to_board(best_move, color)

    # toggle to_move if present
    if hasattr(result_state, 'to_move'):
        result_state.to_move = opponent(color)

    return result_state, win

