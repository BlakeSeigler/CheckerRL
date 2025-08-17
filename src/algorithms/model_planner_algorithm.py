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

    # helpers
    def in_bounds(r, c):
        return 0 <= r < N and 0 <= c < N

    def is_opponent(piece, color):
        if piece == 'o':
            return False
        if color == 'w':
            return piece in ('b', 'kb')
        else:
            return piece in ('w', 'kw')

    def is_own(piece, color):
        if piece == 'o':
            return False
        if color == 'w':
            return piece in ('w', 'kw')
        else:
            return piece in ('b', 'kb')

    def is_king(piece):
        return piece in ('kw', 'kb')

    def promote_if_needed(piece, r, color):
        if color == 'w' and r == 0 and piece == 'w':
            return 'kw'
        if color == 'b' and r == 7 and piece == 'b':
            return 'kb'
        return piece

    # Generate all legal moves for color.
    # Moves are represented as lists of positions: [(r0,c0),(r1,c1),...(rk,ck)]
    def generate_legal_moves(board, color):
        moves = []
        any_capture = False

        # direction for normal pieces: white moves up (-1), black moves down (+1)
        forward_dirs = [-1] if color == 'w' else [1]
        king_dirs = [-1, 1]

        def find_jumps_from(r, c, piece, board_snapshot, visited):
            """Return list of jump sequences starting from (r,c) given current snapshot.
               visited is a set of captured positions to avoid double-capturing same piece."""
            results = []

            if is_king(piece):
                directions = king_dirs
            else:
                directions = forward_dirs

            found_any = False
            for dr in directions:
                for dc in (-1, 1):
                    mr, mc = r + dr, c + dc  # midpoint (opponent)
                    lr, lc = r + 2*dr, c + 2*dc  # landing
                    if in_bounds(mr, mc) and in_bounds(lr, lc):
                        mid_piece = board_snapshot[mr][mc]
                        land_piece = board_snapshot[lr][lc]
                        if is_opponent(mid_piece, color) and land_piece == 'o' and (mr, mc) not in visited:
                            # simulate this capture
                            new_snapshot = [row[:] for row in board_snapshot]
                            new_snapshot[r][c] = 'o'
                            new_snapshot[mr][mc] = 'o'
                            new_snapshot[lr][lc] = piece  # keep piece type for now; promote later at end of move
                            new_visited = set(visited)
                            new_visited.add((mr, mc))
                            # continue searching for multi-captures from landing
                            deeper = find_jumps_from(lr, lc, piece, new_snapshot, new_visited)
                            if deeper:
                                for seq in deeper:
                                    results.append([(r, c), (lr, lc)] + seq[1:])  # chain sequences
                            else:
                                results.append([(r, c), (lr, lc)])
                            found_any = True
            return results

        for r in range(N):
            for c in range(N):
                piece = board[r][c]
                if not is_own(piece, color):
                    continue

                # jumps
                jumps = find_jumps_from(r, c, piece, board, set())
                if jumps:
                    any_capture = True
                    for seq in jumps:
                        moves.append(seq)
                    continue  # if capture exists, we don't add simple slides for this piece

        if any_capture:
            # if any capture exists anywhere, captures are mandatory: return only captures
            return moves

        # no captures found — add simple slides
        for r in range(N):
            for c in range(N):
                piece = board[r][c]
                if not is_own(piece, color):
                    continue
                dirs = king_dirs if is_king(piece) else forward_dirs
                for dr in dirs:
                    for dc in (-1, 1):
                        nr, nc = r + dr, c + dc
                        if in_bounds(nr, nc) and board[nr][nc] == 'o':
                            moves.append([(r, c), (nr, nc)])
        return moves

    # Apply a move sequence to a board and return a new board deep-copied
    def apply_move_to_board(board, move_seq, color):
        newb = [row[:] for row in board]
        r0, c0 = move_seq[0]
        piece = newb[r0][c0]
        newb[r0][c0] = 'o'
        # if it's a single slide
        if len(move_seq) == 2 and abs(move_seq[1][0] - r0) == 1:
            r1, c1 = move_seq[1]
            piece_after = promote_if_needed(piece if piece in ('w', 'b') else piece, r1, color)
            newb[r1][c1] = piece_after
        else:
            # series of jumps; remove captured pieces
            curr = (r0, c0)
            for nxt in move_seq[1:]:
                rN, cN = nxt
                # captured piece is midpoint
                mr, mc = (curr[0] + rN) // 2, (curr[1] + cN) // 2
                newb[mr][mc] = 'o'
                curr = nxt
            # place piece in final landing
            rf, cf = move_seq[-1]
            piece_after = promote_if_needed(piece if piece in ('w', 'b') else piece, rf, color)
            newb[rf][cf] = piece_after
        return newb

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

    legal_moves = generate_legal_moves(board, color)
    if not legal_moves:
        # no legal moves => return deep-copy without change
        return copy.deepcopy(current_state)

    best_move = None
    best_score = -1e9

    for move in legal_moves:
        # simulate our move
        next_board = apply_move_to_board(board, move, color)
        # evaluate after opponent best response (1 ply opponent)
        opp_moves = generate_legal_moves(next_board, opponent(color)) 
        if not opp_moves:
            # opponent has no moves -> we win: large score
            score = 1e6 + evaluate_board(next_board)
            win = True
        else:
            opp_best = 1e9
            for opp_move in opp_moves:
                opp_board = apply_move_to_board(next_board, opp_move, opponent(color))
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
    result_obj.state = apply_move_to_board(board, best_move, color)
    # if your StateVector has a 'to_move' attribute, toggle it
    if hasattr(result_obj, 'to_move'):
        result_obj.to_move = opponent(color)
    return result_obj, win

