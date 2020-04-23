from enum import Enum
from typing import Callable


class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Node(object):

    def __init__(self, id_: int, type_: NodeType, activation_function: Callable[[float], float]) -> None:
        self.id: int = id_
        self.node_type: NodeType = type_
        self.activation_function: Callable[[float], float] = activation_function
