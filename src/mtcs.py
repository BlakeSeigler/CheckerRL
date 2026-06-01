from unittest import skip
import torch
from data import format_data
from nn import NN
import random, math
import copy
import numpy as np

class MTCS:
    """
    Note for values:
    1 is for white
    -1 is for black
    
    """
    def __init__(self, Game, NeuralNetwork):
        self.root = Node(game=Game, color="WHITE", parent=None, change=(None, None))
        self.net = NeuralNetwork
        self.num_simulations = 10000
        self.TEMP_CHANGE_MOVE = 30

    def initialize_tree(self, game):
        """
        Maybe this should just be in init but whateva
        """
        for move in game.get_valid_game_moves("WHITE"): #TODO this needs to be all the moves for all the pieces
            from_row, from_col, to_row, to_col, skipped = move
            game.make_move(from_row, from_col, to_row, to_col, skipped)
            leaf = Node(state=copy.copy(game), color="RED", parent=self.root, change=((from_row, from_col), (to_row, to_col)), skipped=skipped)
            self.root.leaves.append(leaf)

        for leaf in self.root.leaves:
            value = self.simulation(leaf)
            leaf.update_value(value)
            leaf.update_visits()
        
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
            current_node.backpropagation(value)
        else:
            scores = [(node, self.calculate_UBT(node)) for node in current_node.leaves]
            best_node = max(scores, key=lambda x: x[1])[0]
            self.selection(best_node)

    def expansion(self, node):
        """Expands the node by adding new child nodes"""
        num_of_unexpanded = len(node.unexpanded_moves)
        new_leaf = node.unexpanded_moves.pop(random.randint(0, num_of_unexpanded - 1))
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
            state = current_node.get_state()
            (from_row, from_col) = state.change[1]
            moves =  state.board.get_valid_piece_moves(state.board[from_row*8 + from_col])

            # Pick a move at random and make a new node
            move = random.choice(list(moves.items()))
            (to_row, to_col), skipped = move
            state.make_move(from_row, from_col, to_row, to_col, skipped)
            change = ((from_row, from_col), (to_row, to_col))
            current_node = Node(state=copy.copy(state), color=state.turn, parent=current_node, change=change, skipped=skipped)

        
        value = 1 if current_node.get_state().winner() == node.get_turn() else -1
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
            parent.update_value(value)
            parent.update_visits()

    def calculate_UBT(self, node):
        encoded_state = format_data(node.get_state())
        value = self.net.value_forward(encoded_state)
        selection = self.net.selection_forward(encoded_state)
        
        output = value + selection * math.sqrt(math.log(node.parent.visits) / node.visits)  # not 100% sure if N is total parent or total overall
        return output

    def predict(self, game):
        
        # Build out the tree 
        for i in range(self.num_simulations):
            self.selection(game)
        
        for node in self.root.leaves:
            if node.get_value() > best_value:
                best_value = node.get_value()
                best_node = node
        
        (from_row, from_col), (to_row, to_col) = best_node.change
        return from_row, from_col, to_row, to_col, best_node.skipped

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
    def __init__(self, game, color, parent, change, skipped: list):
        self.turn = color
        self.state = game
        self.unexpanded_moves = game.get_valid_game_moves(self.turn) 
        self.values = []
        self.visits = 0
        self.leaves = []
        self.parent = None
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
