from common import StateVector, Move

class Node:

    state: StateVector
    children: list[Move]

    def __init__(self, state: StateVector):
        pass