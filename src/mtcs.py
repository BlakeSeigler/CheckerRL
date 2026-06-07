from unittest import skip
import torch
from data import format_data
from nn import NN
import random, math
import copy
import numpy as np
from checkers.constants import RED, WHITE
from checkers.game import Game
from checkers.piece import Piece

class MTCS:
    """
    Note for values:
    1 is for white
    -1 is for black
    
    """
    def __init__(self, Game, NeuralNetwork):
        self.root = Node(game=Game, color=WHITE, parent=None, change=(None, None), skipped=[])
        self.net = NeuralNetwork
        self.num_simulations = 10000
        self.TEMP_CHANGE_MOVE = 30

    def initialize_tree(self, game):
        """
        Maybe this should just be in init but whateva
        """
        for (from_row, from_col), piece_moves in game.get_valid_game_moves(WHITE):
            for (to_row, to_col), skipped in piece_moves.items():
                this_game = copy.copy(game)
                this_game.make_move(from_row, from_col, to_row, to_col, skipped)
                leaf = Node(game=this_game, color=RED, parent=self.root, change=((from_row, from_col), (to_row, to_col)), skipped=skipped)
                self.root.leaves.append(leaf)

        for leaf in self.root.leaves:
            self.simulation(leaf)
        
        return


    def selection(self, game):
        """
        Uses the UCT with NN to select the best node, stops at terminal or unexplored nodes
        
        I actually think this function is implemented wrong and does the build out when that should be seperate...whateva
        """
        current_node = self.root

        if current_node.unexpanded_moves != []:  # if not fully expanded
            new_node = self.expansion(current_node)
            result = self.simulation(new_node)
            self.backpropagation(new_node, result)
        
        if current_node.leaves == []:
            value = current_node.get_value()
            current_node.update_visits()
            self.backpropagation(current_node, value)
        else:
            nodes_and_scores = [(node, self.calculate_UBT(node)) for node in current_node.leaves]
            best_node = max(nodes_and_scores, key=lambda x: x[1])[0]
            self.selection(best_node)

    def expansion(self, node):
        """Expands the node by adding new child nodes"""
        num_of_unexpanded = len(node.unexpanded_moves)

        selected=False
        while selected == False:
            try:   #TODO still working on this here. Do these bottom two lines work properly? I feel like they don't
                (new_leaf_from_row, new_leaf_from_col), piece_moves = node.unexpanded_moves[(random.randint(0, num_of_unexpanded - 1))]                            # picks a random piece
                (new_leaf_to_row, new_leaf_to_col), new_leaf_skipped = random.choice(list(piece_moves.items()))       
                selected=True
            except: 
                continue             
        
        # with move, make the game
        game = copy.copy(node.state)
        game.make_move(new_leaf_from_row, new_leaf_from_col, new_leaf_to_row, new_leaf_to_col, skipped=new_leaf_skipped)
        new_leaf = Node(game=game, color=game.turn, parent=node, change=((new_leaf_from_row, new_leaf_from_col), (new_leaf_to_row, new_leaf_to_col)), skipped=new_leaf_skipped)
        node.leaves.append(new_leaf)
        return new_leaf

    def simulation(self, node):
        """
        rollouts -- random playouts
        takes in a node and returns the value of the node after going all the way down
        returns the value of a terminal node based on the current nodes turn.
        """
        current_node = node
        while current_node.get_state().winner() == None:

            # Get the moves from the new location
            # Pick a move at random and make a new node
            pieces = current_node.get_state().get_valid_pieces(current_node.get_state().turn)
            move = None
            while move == None:         #TODO Does this while loop leave open the possibility of a stalemate looping this?
                piece = random.choice(pieces)
                moves =  current_node.get_state().board.get_valid_piece_moves(piece) 
                try:
                    move = random.choice(list(moves.items()))
                except: 
                    continue

            (to_row, to_col), skipped = move
            from_row, from_col = piece.row, piece.col
            new_state = copy.copy(current_node.get_state())
            new_state.make_move(from_row, from_col, to_row, to_col, skipped)
            change = ((from_row, from_col), (to_row, to_col))
            current_node = Node(game=new_state, color=new_state.turn, parent=current_node, change=change, skipped=skipped)

        value = 1 if current_node.get_state().winner() == node.get_turn() else -1 #TODO also there is no end here if the game just goes on forever, might need to cap and draw -- that could also lead to a highly sparse endgame so you might need to create some incentive in some way
        node.update_value(value)
        node.update_visits()
        return value


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

    def calculate_UBT(self, node):
        encoded_state = format_data(node.state, node.turn)
        value = self.net.value_forward(encoded_state)
        selection = self.net.selection_forward(encoded_state)
        
        output = value + selection * math.sqrt(math.log(node.parent.visits) / node.visits)  # not 100% sure if N is total parent or total overall
        return output

    def predict(self, game):
        
        # Build out the tree 
        for i in range(self.num_simulations):
            self.selection(game)
        
        best_value = -1e9
        for node in self.root.leaves:
            if node.get_value() > best_value:
                best_value = node.get_value()
                best_node = node
        
        (from_row, from_col), (to_row, to_col) = best_node.change
        return (from_row, from_col), (to_row, to_col), best_node.skipped

    def calculate_distribution(self, game, move_number: int):
        """
        Policy distribution.  
        """

        # first get the distribution
        distribution = np.zeros(4096)
        for leaf in self.root.leaves:
            from_idx = leaf.change[0][0] * 8 + leaf.change[0][1]
            to_idx = leaf.change[1][0] * 8 + leaf.change[1][1]
            distribution[from_idx * 64 + to_idx] = leaf.visits

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
        self.values = []
        self.visits = 0
        self.leaves = []
        self.parent = parent
        self.change: tuple[tuple[int, int]] = change # the move that changed
        self.skipped = skipped

    def get_value(self):
        return sum(self.values) / len(self.values)

    def update_value(self, value):
        self.values.append(value)

    def update_visits(self):
        self.visits += 1

    def get_state(self):
        return self.state
        
    def get_turn(self):
        return self.turn

    def expand(self, move):
        self.leaves.append(move)
        self.unexpanded_moves.remove(move)
