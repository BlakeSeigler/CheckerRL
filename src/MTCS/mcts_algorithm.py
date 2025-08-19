"""
Monte Carlo Tree Search Implementation for checkers strategy
- The MAB heurisitic for traversing the tree is Upper Confidence Bound 1
- The heuristic for the simulation phase will be a NN similar to that done be AlphaGO
    > I hate the idea of doing random moves and a fixed heuristic feels too inefficient to me
"""

from common.src.common import StateVector

def get_checkers_move(current_state: StateVector, self_color: str, i = 5):
    """
    MCTS Algorithm with phases, Selection, Expansion, Simulation, Iteration
    """
    ## helpers
    def calculateUCB1():
        pass

    
    # -----------------


    # Generate Moves
    moves = current_state.generate_legal_moves(current_state.state, self_color)
    terminating = current_state.terminating

    # Selection Stage - Calculate MAB problem for nodes
    if not terminating:
    for move in moves:
        calculateUCB1(move, self_color)

        