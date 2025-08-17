from common import Player, StateVector
from algorithms import model_planner_algorithm

def Run_Game() -> Player | None:
    """Run game and return the winner"""
    print("Starting Game")
    white = Player('w', model_planner_algorithm.get_checkers_move)
    black = Player('b', model_planner_algorithm.get_checkers_move)
    state = StateVector()
    timeout = 400

    move: int = 1
    print("Making first move")
    while not white.get_is_winner() and not black.get_is_winner() and timeout > move:
        if move % 2 == 1:
            state = white.get_move(state)
        elif move % 2 == 0:
            state = black.get_move(state)

        print(f"Made move {move}", "\n", state)
    
        move += 1

        if white.get_is_winner():
            return white
        elif black.get_is_winner():
            return black
    
    return None

if __name__=="__main__":
    Run_Game()