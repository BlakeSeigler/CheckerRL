import numpy as np

class Move:
    start: tuple[int]
    end:tuple[int]     
    captured: list[tuple[int]]  

    def __init__(self, start, end, captured: None | list[tuple[int]] = None):
        self.start = start
        self.end = end
        
        if captured != None:
            self.captured = captured
        else:
            self.captured = []


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
    
    def get_legal_moves(self, color: str) -> list[Move]:
        """Return a list of Move objects for `color` on the current self.state board.
           Captures are mandatory; if any capture exists, only capture moves are returned.
        """
        board = self.state
        moves: list[Move] = []
        any_capture = False

        forward_dirs = [-1] if color == 'w' else [1]
        king_dirs = [-1, 1]

        def in_bounds(r, c):
            return 0 <= r < 8 and 0 <= c < 8

        def is_opponent(piece: str, color: str):
            if piece == 'o':
                return False
            if color == 'w':
                return piece in ('b', 'kb')
            else:
                return piece in ('w', 'kw')

        def is_own(piece: str, color: str):
            if piece == 'o':
                return False
            if color == 'w':
                return piece in ('w', 'kw')
            else:
                return piece in ('b', 'kb')

        def is_king(piece: str):
            return piece in ('kw', 'kb')

        # returns list of position-sequences (list of (r,c)) representing jump chains
        def find_jumps_from(r, c, piece, board_snapshot, visited):
            results = []
            directions = king_dirs if is_king(piece) else forward_dirs

            for dr in directions:
                for dc in (-1, 1):
                    mr, mc = r + dr, c + dc        # midpoint (opponent)
                    lr, lc = r + 2*dr, c + 2*dc   # landing
                    if in_bounds(mr, mc) and in_bounds(lr, lc):
                        mid_piece = board_snapshot[mr][mc]
                        land_piece = board_snapshot[lr][lc]
                        if is_opponent(mid_piece, color) and land_piece == 'o' and (mr, mc) not in visited:
                            # simulate capture on snapshot
                            new_snapshot = [row[:] for row in board_snapshot]
                            new_snapshot[r][c] = 'o'
                            new_snapshot[mr][mc] = 'o'
                            new_snapshot[lr][lc] = piece  # keep piece type for further jumps
                            new_visited = set(visited)
                            new_visited.add((mr, mc))
                            # find deeper jumps from landing square
                            deeper = find_jumps_from(lr, lc, piece, new_snapshot, new_visited)
                            if deeper:
                                for seq in deeper:
                                    results.append([(r, c), (lr, lc)] + seq[1:])
                            else:
                                results.append([(r, c), (lr, lc)])
            return results

        # first collect all jump sequences (captures)
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if not is_own(piece, color):
                    continue
                jumps = find_jumps_from(r, c, piece, board, set())
                if jumps:
                    any_capture = True
                    for seq in jumps:
                        # convert seq -> Move with captured midpoints
                        captured = []
                        for i in range(len(seq) - 1):
                            sr, sc = seq[i]
                            er, ec = seq[i+1]
                            mr, mc = (sr + er)//2, (sc + ec)//2
                            captured.append((mr, mc))
                        moves.append(Move(start=seq[0], end=seq[-1], captured=captured))
        if any_capture:
            return moves  # only captures allowed

        # no captures found -> add simple slides
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
                            moves.append(Move(start=(r, c), end=(nr, nc), captured=[]))
        return moves


    def apply_move_to_board(self, move: Move, color: str) -> "StateVector":
        """Apply a Move to the current self.state and return a NEW StateVector (deep copy).
           The original StateVector is left unchanged.
        """
        def promote_if_needed(piece: str, r: int, color: str) -> str:
            if piece == 'w' and color == 'w' and r == 0:
                return 'kw'
            if piece == 'b' and color == 'b' and r == 7:
                return 'kb'
            return piece

        # deep copy board
        newb = [row[:] for row in self.state]
        sr, sc = move.start
        piece = newb[sr][sc]
        # remove from start
        newb[sr][sc] = 'o'

        # if a capture move (captured list non-empty) remove captured midpoints
        if move.captured:
            # remove captured pieces
            for (cr, cc) in move.captured:
                newb[cr][cc] = 'o'
            # place piece at final location and handle promotion
            fr, fc = move.end
            # if the piece was already a king, keep it; otherwise promote if needed
            piece_to_place = piece
            if piece in ('w', 'b'):
                piece_to_place = promote_if_needed(piece, fr, color)
            newb[fr][fc] = piece_to_place
        else:
            # simple slide
            fr, fc = move.end
            piece_to_place = piece
            if piece in ('w', 'b'):
                piece_to_place = promote_if_needed(piece, fr, color)
            newb[fr][fc] = piece_to_place

        # build new StateVector and return
        new_sv = StateVector()
        new_sv.state = newb
        new_sv.terminating = False
        # char_dict and inv_char_dict already set in __init__, so no change needed
        return new_sv