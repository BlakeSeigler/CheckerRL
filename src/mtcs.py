import torch
from data import format_data
from nn import NN

class MTCS:
    """
    Note for values:
    1 is for white
    -1 is for black
    
    """
    def __init__(self, Game, NeuralNetwork):
        self.root = Node(game=Game, color="WHITE", parent=None, change=None)
        self.net = NeuralNetwork
        self.num_simulations = 10000

    def initialize_tree(self, game):
        moves = []
        for piece in game.get_valid_pieces("WHITE"):
            for move in game.get_valid_moves(piece):
                game.make_move(move)
                leaf = Node(state=game, color="BLACK", parent=self.root, change=move)
                self.root.leaves.append(leaf)

        for leaf in self.root.leaves:
            value = self.simulation(leaf)
            leaf.update_value(value)
            leaf.update_visits()
        
        return


    def selection(self, game):
        """
        Uses the UCT with NN to select the best node, stops at terminal or unexplored nodes
        
        I actually think this function is implemented wrong and does the build out when taht should be seperate...whateva
        """
        current_node = self.root

        if current_node.unexpanded_moves != []  # if not fully expanded
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
            state = current_node.get_state()
            moves = state.get_valid_moves(state.turn)
            move = moves[random.randint(0, len(moves) - 1)]
            state = state.make_move(move)
            current_node = Node(state=state, color=state.turn, parent=current_node)
        value = 1 if current_node.get_state().winner() == "WHITE" else -1
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

    def calculate_UBT(self, node: Node):
        encoded_state = format_data(node.get_state())
        value = self.net.value_forward(encoded_state)
        selection = self.net.selection_forward(encoded_state)
        
        output = value + selection * math.sqrt(math.log(node.parent.visits) / node.visits)  # not 100% sure if N is total parent or total overall
        return output

    def predict(self, game):
        
        # Build out the tree 
        for i in range(self.num_simulations):
            selected_node = self.selection(game)
        
        for node in self.root.leaves:
            if node.get_value() > best_value:
                best_value = node.get_value()
                best_node = node
        
        return best_node.change

class Node:
    def __init__(self, game, color, parent, change):
        self.turn = color
        self.state = game
        self.unexpanded_moves = game.get_valid_moves(self.turn) 
        self.values = []
        self.visits = 0
        self.leaves = []
        self.parent = None
        self.change = change # the move that changed

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

    def expand(self, move: Node):
        self.leaves.append(move)
        self.unexpanded_moves.remove(move)
