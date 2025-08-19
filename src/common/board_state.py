import numpy as np

class StateVector:

    state: list[list[str]]

    def __init__(self):
        self.state = [
            ['o', 'b', 'o', 'b', 'o', 'b', 'o', 'b'],
            ['b', 'o', 'b', 'o', 'b', 'o', 'b', 'o'],
            ['o', 'b', 'o', 'b', 'o', 'b', 'o', 'b'],
            ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
            ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
            ['w', 'o', 'w', 'o', 'w', 'o', 'w', 'o'],
            ['o', 'w', 'o', 'w', 'o', 'w', 'o', 'w'],
            ['w', 'o', 'w', 'o', 'w', 'o', 'w', 'o'],
        ]
        self.char_dict = {
            'o': 0.0,
            'w': 1.0, 
            'b': -1.0, 
            'kw': 2.0, 
            'kb': -2.0, 
        }
        self.inv_char_dict = {v: k for k, v in self.char_dict.items()}

    @classmethod
    def from_char_numpy_vector(self, state: np.ndarray[str]) -> bool:
        assert state.shape == (64,)
        try:
            self.state = state.reshape(8,8)
            return True
        except:
            return False

    def to_char_numpy_array(self) -> np.ndarray[str]:
        return np.array(self.state).flatten()
    
    @classmethod
    def from_float_numpy_vector(self, state: np.ndarray[float]) -> bool:
        assert state.shape == (64,)
        try:
            rounded = np.round(np.array(state))
            intermediate = np.array([self.inv_char_dict[f] for f in rounded])     
            self.state = intermediate.reshape(8,8)
            return True
        except:
            return False

    def to_int_numpy_array(self) -> np.ndarray[float]:
        intermediate = np.array(self.state).flatten()
        return np.array([self.char_dict[f] for f in intermediate])
    
    def __str__(self):
        text = ""
        for line in self.state:
            text += f"{line}" + "\n "

        return text
    
    def generate_legal_moves(board, color):
        moves = []
        any_capture = False

        # direction for normal pieces: white moves up (-1), black moves down (+1)
        forward_dirs = [-1] if color == 'w' else [1]
        king_dirs = [-1, 1]

        # helpers
        def in_bounds(r, c):
            return 0 <= r < 8 and 0 <= c < 8

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
        
        def opponent(color):
            return 'b' if color == 'w' else 'w'

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

        for r in range(8):
            for c in range(8):
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
        for r in range(8):
            for c in range(8):
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

        # HELPERS
        def promote_if_needed(piece, r, color):
            if color == 'w' and r == 0 and piece == 'w':
                return 'kw'
            if color == 'b' and r == 7 and piece == 'b':
                return 'kb'
            return piece
            

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