from src.board_state import StateVector

class Player():

    def __init__(self, color: str):
        self.win = False
        self.color = color

    def get_move(self, current_state: StateVector) -> StateVector:

        #TODO Logic for getting a new move
        new_state = current_state
        self._check_valid_state_change(current_state, new_state)

        return new_state

    def get_is_winner(self) -> bool:
        return self.win
    
    def _set_as_winner(self):
        self.win = True

    def _check_valid_state_change(self, current_state: StateVector, new_state: StateVector) -> bool:
        def is_own_piece(piece: str) -> bool:
            return piece.lower().startswith(self.color)

        def is_opponent_piece(piece: str) -> bool:
            opponent_color = 'b' if self.color == 'w' else 'w'
            return piece.lower().startswith(opponent_color)

        # collect changed squares
        changed_squares = []
        for r in range(8):
            for c in range(8):
                curr_piece = current_state.state[r][c]
                new_piece = new_state.state[r][c]
                if curr_piece != new_piece:
                    changed_squares.append((r, c, curr_piece, new_piece))

        # classify changes:
        # - moved_from: own piece -> empty (exactly one)
        # - moved_to: empty -> own piece (exactly one)
        # - captured_pieces: opponent piece -> empty (0 or more)
        moved_from = None
        moved_to = None
        captured_positions = []  # list of (r,c,piece)

        for (r, c, curr_piece, new_piece) in changed_squares:
            if curr_piece != 'o' and new_piece == 'o':
                # piece removed
                if is_own_piece(curr_piece):
                    if moved_from is not None:
                        return False
                    moved_from = (r, c, curr_piece)
                elif is_opponent_piece(curr_piece):
                    captured_positions.append((r, c, curr_piece))
                else:
                    return False
            elif curr_piece == 'o' and new_piece != 'o':
                # piece placed
                if is_own_piece(new_piece):
                    if moved_to is not None:
                        return False
                    moved_to = (r, c, new_piece)
                else:
                    return False
            else:
                # any other kind of change (e.g. piece swapped types in place) is invalid
                return False

        # must have exactly one moved-from and one moved-to
        if moved_from is None or moved_to is None:
            return False

        fr, fc, from_piece = moved_from
        tr, tc, to_piece = moved_to

        from_piece_lower = from_piece.lower()
        to_piece_lower = to_piece.lower()
        is_king = from_piece_lower in ('kw', 'kb')

        # forward direction per your request: white -> +1 (down the array), black -> -1
        forward_direction = 1 if self.color == 'w' else -1
        last_row = 7 if forward_direction == 1 else 0

        # PROMOTION RULE:
        # - If the resulting piece is a king (to_piece is 'kw' or 'kb'), it must be because
        #   the piece landed on the far rank for that color.
        # - If a non-king ends on the far rank but wasn't promoted in new_state, reject.
        if to_piece_lower in ('kw', 'kb'):
            # check the king type matches color
            expected_king = 'kw' if self.color == 'w' else 'kb'
            if to_piece_lower != expected_king:
                return False
            # must land on last row to be promoted
            if tr != last_row:
                return False
        else:
            # if landed on last row but wasn't promoted, invalid
            if (not is_king) and (tr == last_row):
                return False

        # ensure moved piece belongs to player (safety check)
        if not is_own_piece(from_piece):
            return False

        # SIMPLE MOVE (no captures)
        if len(captured_positions) == 0:
            if abs(tr - fr) == 1 and abs(tc - fc) == 1:
                # direction check for non-kings
                if is_king or (tr - fr) == forward_direction:
                    return True
                else:
                    return False
            else:
                return False

        # CAPTURE(S) - one or more captured pieces allowed (chain)
        # We must verify there exists a sequence of 1-step captures (each hop is 2 squares diagonal)
        # that starts at (fr,fc), jumps over each captured position exactly once, and ends at (tr,tc).
        cap_set = {(r, c) for (r, c, _) in captured_positions}

        # helper: DFS search for an ordering of captured pieces that yields final position
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dfs(rpos: int, cpos: int, remaining_caps_frozenset: tuple) -> bool:
            remaining_caps = set(remaining_caps_frozenset)
            # if no remaining captured pieces, we must be at the final landing
            if not remaining_caps:
                return (rpos, cpos) == (tr, tc)

            # try jumping over each remaining captured piece if it is adjacent-diagonal
            for cap in list(remaining_caps):
                cap_r, cap_c = cap
                dir_r = cap_r - rpos
                dir_c = cap_c - cpos
                # cap must be one diagonal step away
                if abs(dir_r) == 1 and abs(dir_c) == 1:
                    landing_r = rpos + 2 * dir_r
                    landing_c = cpos + 2 * dir_c
                    # landing must be on-board
                    if not (0 <= landing_r < 8 and 0 <= landing_c < 8):
                        continue
                    # landing square must be empty in the current_state (since piece only occupies final square in new_state)
                    # It's allowed that landing == final destination (tr,tc) which we already know changed to our piece
                    if (landing_r, landing_c) != (tr, tc) and current_state.state[landing_r][landing_c] != 'o':
                        continue
                    # direction check for non-king pieces: each hop must be forward
                    if (not is_king) and (dir_r != forward_direction):
                        continue
                    # recursively try with that capture removed
                    new_remaining = tuple(sorted(remaining_caps - {cap}))
                    if dfs(landing_r, landing_c, new_remaining):
                        return True
            return False

        # run DFS starting from moved-from location
        remaining_caps_tuple = tuple(sorted(cap_set))
        return dfs(fr, fc, remaining_caps_tuple)
