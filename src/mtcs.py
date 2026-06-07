from data import format_data
import math
import copy
import numpy as np
from tqdm import tqdm
from checkers.constants import WHITE

class MTCS:
    """
    Note for values:
    1 is for white
    -1 is for black
    
    """
    def __init__(self, Game, NeuralNetwork):
        self.root = Node(game=Game, color=WHITE, parent=None, change=(None, None), skipped=[])
        self.net = NeuralNetwork
        self.num_simulations = 500
        self.TEMP_CHANGE_MOVE = 30
        self.exploratory_term = 1                       # a term that controls exploration weight 
        
        # Initialize the first few nodes
        self.simulation(self.root)


    def selection(self, current_node):
        """
        Starting at root, go through the tree until we get to an...
        - an unexpanded node
        - a terminal node
        if unexpanded, run the sim on it and backpropogate up. If terminal, then just backpropogate up
        """

        if current_node.terminal is True:           # node is terminal
            current_node.update_visits()
            self.backpropagation(current_node, 1 if current_node.turn == current_node.state.winner() else -1)

        elif current_node.expanded is False:        # if not expanded/simmed, expand/sim and backpropogate up
            self.simulation(current_node)
            current_node.update_visits()
            self.backpropagation(current_node, current_node.get_value())
            current_node.expanded = True

        else:               # expanded/simmed and not terminal, go down a child branch and select again
            PUBT_weights = self.calculate_PUBT_weights(current_node)
            best_node = max(PUBT_weights, key=lambda x: x[1])[0]
            return self.selection(best_node)

    def calculate_PUBT_weights(self, node) -> list[tuple]:
        outputs=[]
        priors = node.get_priors()
        total_visits = node.visits
        
        for child in node.children: # make sure this works for expanded and unexpanded properly. 
            (fx, fy), (tx, ty) = child.change
            value = child.get_value()
            prior = priors[(fx*8+fy)*64 + (tx*8 + ty)]
            outputs.append((child, value + self.exploratory_term * prior * math.sqrt(math.log(total_visits) / (1 + child.visits)))) #TODO what happens when this is 0?
        
        return outputs  # returns a list of (node, output)

    def simulation(self, node):
        """
        uses the NN to get a value for the position and updates visits -> unless the node is terminal then the value is known.
        """
        game = node.state
        color = game.turn
        data = format_data(game, color)

        # calculate values
        value, priors = self.net.forward(data)

        # set the nodes values
        node.expand()
        node.values.append(value)
        node.set_priors(priors)

        return


    def backpropagation(self, node, value):
        """
        Takes in a node and backpropogates all the way up starting at the immediate parent
        """
        parent = node.parent
        if parent != None:
            parent.update_value(value)
            parent.update_visits()
            parent.get_value()
            self.backpropagation(parent, value)
        else:
            return

    def predict(self):
        
        # Build out the tree
        for i in tqdm(range(self.num_simulations), desc="MCTS simulations", leave=False):
            self.selection(self.root)

        # Select the best move -- this is the move with highest N -- high value and trustworthiness or both baked into N
        best_N = 0
        best_node = None
        for child in self.root.children:
            if child.visits > best_N:
                best_node = child
                best_N = child.visits
        
        (from_row, from_col), (to_row, to_col) = best_node.change
        return (from_row, from_col), (to_row, to_col), best_node.skipped

    def calculate_distribution(self, game, move_number: int):
        """
        Policy distribution.  
        """

        # first get the distribution
        distribution = np.zeros(4096)
        for child in self.root.children:
            from_idx = child.change[0][0] * 8 + child.change[0][1]
            to_idx = child.change[1][0] * 8 + child.change[1][1]
            distribution[from_idx * 64 + to_idx] = child.visits

        # change the distribution for both high and low temperature
        if move_number >= self.TEMP_CHANGE_MOVE:    # temperature of 1
            distribution = distribution / distribution.sum()
            return distribution
        else:   # temperature of 0
            new_distribution = np.zeros(4096, dtype=np.float32)
            new_distribution[np.argmax(distribution)] = 1.0
            return new_distribution




class Node:
    def __init__(self, game, color, parent, change, skipped: list=None):
        self.turn = color
        self.state = game
        self.unexpanded_moves = game.get_valid_game_moves(self.turn)  # a list of ( (from_row, from_col), {(to_row, to_col): [skipped pieces]} )
        self.values = [0]
        self.visits = 0
        self.children = None
        self.parent = parent
        self.change: tuple[tuple[int, int]] = change # the move that changed
        self.skipped = skipped
        self.priors = None
        self.expanded=False
        self.terminal=False

        if game.winner():    # TODO doesn't account for draw
            self.terminal=True

    def get_value(self):
        return sum(self.values) / len(self.values)

    def update_value(self, value):
        self.values.append(value)

    def update_visits(self):
        self.visits += 1

    def set_priors(self, prior):
        self.priors = prior

    def get_priors(self):
        return self.priors

    def expand(self):
        self.children = []

        for move in self.unexpanded_moves:
            (fx, fy), destinations = move
        
            for (tx, ty), skipped in destinations.items():
                new_game = copy.deepcopy(self.state)
                new_game.make_move(fx, fy, tx, ty, skipped)
                self.children.append(Node(
                    game=new_game,
                    color=new_game.turn,
                    parent=self,
                    change=((fx, fy), (tx, ty)),
                    skipped=skipped,
                ))
                