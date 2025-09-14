from common import StateVector, Move
from __future__ import annotations
import random
import torch

class Node:

    c: int = ...
    state: StateVector
    visits: int
    wins: int
    is_expanded: bool
    terminal: bool
    parent: Node
    unsimulated_children: list[Node]
    simulated_children: list[Node]

    def __init__(self, state: StateVector):
        self.wins = 0
        self.visits = 0
        self.unsimulated_children = None
        self.state = state
        self.is_expanded = False
        self.is_terminal = False
        self.simulated_children = []

        self.unsimulated_children = self.get_children()

    def expand_node(self) -> Node:
        """
        Expands out one new child node.
        """
        if len(self.unsimulated_children) + len(self.simulated_children) == 0:
            self.set_terminal()
        else:
            new_child_index = random.randint(0, len(self.unsimulated_children) - 1)         # random picking
            new_child = self.unsimulated_children.pop(new_child_index)
            self.simulated_children.append(new_child)

        return new_child
    

    def get_UCB1(self, n):

        val = torch.argmax(
            self.reward + self.c * torch.sqrt( torch.log(n) / self.visits)
        )

        return val
    
    def get_children():
        ## TODO Define legal moves method and call for states
        pass
    
    def set_current_reward(self, reward):
        self.reward = reward

    def get_current_reward(self):
        return self.reward
    
    def get_is_terminal(self) -> bool:
        return self.terminal

    def get_children(self):
        return self.children
    
    def set_terminal(self):
        self.terminal = True