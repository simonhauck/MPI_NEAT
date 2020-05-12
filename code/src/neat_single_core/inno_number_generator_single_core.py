from typing import Union

from neat_core.models.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.models.node import Node


class InnovationNumberGeneratorSingleCore(InnovationNumberGeneratorInterface):

    def __init__(self) -> None:
        self.node_counter = 0
        self.connection_counter = 0

        self.node_innovations = []
        self.connection_innovations = []

    def next_generation(self, generation: int) -> None:
        super().next_generation(generation)

    def get_node_innovation_number(self, node1: Node = None, node2: Node = None) -> Union[int, str]:
        if node1 is None or node2 is None:
            return self._get_new_node_number()

        for n1_innovation, n2_innovation, innovation in self.node_innovations:
            if n1_innovation == node1.innovation_number and n2_innovation == n2_innovation:
                return innovation

        # No stored record found, create a new one
        new_number = self._get_new_node_number()
        self.node_innovations.append((node1.innovation_number, node2.innovation_number, new_number))

        return new_number

    def get_connection_innovation_number(self, input_node: Node = None, output_node: Node = None) -> Union[int, str]:
        if input_node is None or output_node is None:
            return self._get_new_connection_number()

        for n1_innovation, n2_innovation, innovation in self.connection_innovations:
            if n1_innovation == input_node.innovation_number and n2_innovation == output_node.innovation_number:
                return innovation

        # No stored record found, create a new one
        new_number = self._get_new_connection_number()
        self.connection_innovations.append((input_node.innovation_number, output_node.innovation_number, new_number))

        return new_number

    def _get_new_node_number(self) -> Union[int, str]:
        """
        Return the current number from the innovation counter and increment the value
        :return: the current node innovation number
        """
        tmp = self.node_counter
        self.node_counter += 1
        return tmp

    def _get_new_connection_number(self) -> Union[int, str]:
        """
        Return the current number for the connection innovation and increment the value
        :return: an innovation number for connections
        """
        tmp = self.connection_counter
        self.connection_counter += 1
        return tmp
