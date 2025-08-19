from common.src.common.board_state import StateVector

class Player():

    def __init__(self, color: str, strategy):
        self.win = False
        self.color = color
        self.strategy = strategy

    def get_move(self, current_state: StateVector) -> StateVector:

        # Logic for getting a new move
        new_state, win = self.strategy(current_state, self.color)

        if win:
            self.win = True

        return new_state

    def get_is_winner(self) -> bool:
        return self.win
    
    def _set_as_winner(self):
        self.win = True
