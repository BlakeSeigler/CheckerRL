from node import Node
from common import StateVector
from NeuralNetwork import NeuralNetwork

class MCTS_Tree:
    base: Node
    current_root: Node
    current_target: Node
    network: NeuralNetwork

    def __init__(self, network: NeuralNetwork):
        state = StateVector()
        network = network
        self.base = Node(state=state)
        self.current_root = self.base
        self.current_target = self.current_root

    def search(self):
        """Searches the existing tree and gets a node."""

        # Pick a leaf node - defines as current target
        self._selection(self.current_root)

        new_node = self._expansion()

        self._simulation(new_node)

        self._backpropogate(new_node)

    def _selection(self, node: Node) -> None:
        """
        Upper Confidence Bound MAB traversal for MCTS, finds the bottom leaf node
        """
        current_node = node

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

        return self.current_target        

    def _expansion(self) -> bool:

        try:
            self.current_target.expand_node()
            return True
        except:
            return False


    
    def _simulation(self):

        # Once I hit a lead node I will simulate it and store that data
        self.network.predict(self.current_target)
        pass

    def _backpropogate(self):

        # Backpropogate the results up the tree again

        pass