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

        moved_from = None  # (row, col, piece)
        moved_to = None    # (row, col, piece)

        # Detect moved-from and moved-to squares by comparing current and new states
        for r in range(8):
            for c in range(8):
                curr_piece = current_state.state[r][c]
                new_piece = new_state.state[r][c]

                if curr_piece != new_piece:
                    # Piece removed from square
                    if curr_piece != 'o' and new_piece == 'o':
                        if moved_from is not None:
                            # More than one piece moved away - invalid
                            return False
                        moved_from = (r, c, curr_piece)
                    # Piece placed on square
                    elif curr_piece == 'o' and new_piece != 'o':
                        if moved_to is not None:
                            # More than one piece placed - invalid
                            return False
                        moved_to = (r, c, new_piece)
                    else:
                        # Unexpected change (e.g. piece changed into another piece or multiple changes)
                        return False

        # There must be exactly one piece moved
        if moved_from is None or moved_to is None:
            return False

        fr, fc, from_piece = moved_from
        tr, tc, to_piece = moved_to

        # Check the moved piece belongs to the player
        if not is_own_piece(from_piece):
            return False

        # Check that the piece type is preserved or promoted properly
        # For now, allow same piece or promotion (non-king to king)
        allowed_promotions = {
            ('w', 'kw'),
            ('b', 'kb'),
        }
        from_piece_lower = from_piece.lower()
        to_piece_lower = to_piece.lower()
        if from_piece_lower != to_piece_lower:
            # Allow promotion: e.g. 'w' -> 'kw' or 'b' -> 'kb'
            if (self.color, to_piece_lower) not in allowed_promotions:
                return False

        # The destination square must have been empty
        if current_state.state[tr][tc] != 'o':
            return False

        # Movement vector
        row_diff = tr - fr
        col_diff = tc - fc

        is_king = from_piece_lower in ('kw', 'kb')

        # Direction: white moves up (-1), black moves down (+1)
        forward_direction = -1 if self.color == 'w' else 1

        # Simple move: diagonal one step
        if abs(row_diff) == 1 and abs(col_diff) == 1:
            if is_king or row_diff == forward_direction:
                # Ensure no captures: board difference should only be move from-to, no pieces removed
                # Check no pieces removed in jumped squares (none jumped for simple move)
                # Since we already confirmed only one piece moved, this is valid
                return True
            else:
                return False

        # Capture move: diagonal two steps, jumping over opponent piece
        elif abs(row_diff) == 2 and abs(col_diff) == 2:
            jumped_r = fr + row_diff // 2
            jumped_c = fc + col_diff // 2
            jumped_piece = current_state.state[jumped_r][jumped_c]

            # There must be an opponent piece to capture
            if jumped_piece == 'o':
                return False

            if is_own_piece(jumped_piece):
                return False

            # For non-king pieces, direction must be forward
            if not is_king and row_diff != 2 * forward_direction:
                return False

            # Check that the jumped piece was removed in new_state
            if new_state.state[jumped_r][jumped_c] != 'o':
                return False

            return True

        else:
            # Invalid move length or direction
            return False