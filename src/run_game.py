from board_state import StateVector
from player import Player

def Run_Game() -> Player | None:
    """Run game and return the winner"""
    white = Player('w')
    black = Player('b')
    state = StateVector()
    timeout = 400

    move: int = 1
    while not white.get_is_winner() and not black.get_is_winner() and not timeout > move:
        if move % 2 == 1:
            state = white.get_move(state)
        elif move % 2 == 0:
            state = black.get_move(state)

    
    if white.get_is_winner():
        return white
    elif black.get_is_winner():
        return black
    else:
        return None
    
        # Need to include that they both can only move their own states