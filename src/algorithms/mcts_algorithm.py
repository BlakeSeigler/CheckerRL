"""
Monte Carlo Tree Search Implementation for checkers strategy
- The MAB heurisitic for traversing the tree is Upper Confidence Bound 1
- The heuristic for the simulation phase will be a NN similar to that done be AlphaGO
    > I hate the idea of doing random moves and a fixed heuristic feels too inefficient to me
"""

from common import StateVector

def get_checkers_move(current_state: StateVector, self_color: str):
    """
    
    """