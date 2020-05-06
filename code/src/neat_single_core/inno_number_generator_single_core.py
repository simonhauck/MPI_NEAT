from typing import Union

from neat_core.models.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.models.node import Node


class InnovationNumberGeneratorSingleCore(InnovationNumberGeneratorInterface):

    def __init__(self) -> None:
        self.node_counter = 0
        self.connection_counter = 0

    def get_node_innovation_number(self, node1: Node = None, node2: Node = None) -> Union[int, str]:
        """
        Create a new innovation number for each node. The number starts at 0 and will be incremented every request
        :param node1: not used
        :param node2: not used
        :return: the generated innovation number as int
        """
        tmp = self.node_counter
        self.node_counter += 1
        return tmp

    def get_connection_innovation_number(self, input_node: Node = None, output_node: Node = None) -> Union[int, str]:
        """
        Create a new innovation number for each connection. The number starts at 0 and is incremented every time
        :param input_node: not used
        :param output_node:  not used
        :return: the generated innovation number as int
        """
        tmp = self.connection_counter
        self.connection_counter += 1
        return tmp
