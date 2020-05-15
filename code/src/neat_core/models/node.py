from enum import Enum
from typing import Callable
from typing import Union


class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Node(object):

    def __init__(self, innovation_number: Union[int, str], type_: NodeType, bias: float,
                 activation_function: Callable[[float], float],
                 x_position: float) -> None:
        """
        Create a node for a neural network
        :param innovation_number: the assigned innovation number for this node
        :param activation_function: the used activation function
        :param x_position: the x position in the graph
        """
        self.innovation_number: Union[int, str] = innovation_number
        self.node_type: NodeType = type_
        self.bias: float = bias
        self.activation_function: Callable[[float], float] = activation_function
        self.x_position: float = x_position
