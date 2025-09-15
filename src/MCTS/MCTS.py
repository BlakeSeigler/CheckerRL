from node import Node
from common import StateVector
from NeuralNetwork import NeuralNetwork

class MCTS_Tree:
    base: Node
    current_target: Node
    network: NeuralNetwork

    def __init__(self, network: NeuralNetwork):
        state = StateVector()
        network = network
        self.base = Node(state=state)                 # Bottom of the MCTS Tree
        self.current_target = self.base                 # A pseudo tree root

    def search(self):
        """Searches the existing tree and gets a node."""

        # Pick a leaf node - defines as current target
        self.current_target = self._selection(self.current_target)

        if not self.current_target.get_is_terminal() and not self.current_target.get_is_expanded():
            new_node = self._expansion()

        outcome = self._simulation(new_node)

        self._backpropogate(new_node)

    def _selection(self, current_node: Node) -> Node:
        """
        Upper Confidence Bound MAB traversal for MCTS, finds the bottom leaf node
        """

        if current_node.get_is_terminal():
            return current_node

        else:
            while not current_node.get_is_terminal():

                # If the current node has all its kids, use UCB to traverse down
                if current_node.is_expanded:
                    children = current_node.get_children()
                    highest_pick = children[0]
                    for child in children:
                        if child.get_UCB1() > highest_pick.get_UCB1():
                            highest_pick = child 
                
                    current_node = highest_pick
                else:
                    return current_node      

    def _expansion(self) -> Node:

        try:
            new_node = self.current_target.expand_node()
            return new_node
        except:
            raise Exception("Could not expand node for some reason.")


    
    def _simulation(self, node_to_simulate: Node):
        """Given a Node, predict the outcome of the game"""
        outcome = self.network.predict(node_to_simulate)
        return outcome

    def _backpropogate(self):

        # Backpropogate the results up the tree again

        pass