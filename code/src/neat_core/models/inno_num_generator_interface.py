from typing import Union

from neat_core.models.node import Node


class InnovationNumberGeneratorInterface(object):

    def get_node_innovation_number(self, node1: Node = None, node2: Node = None) -> Union[int, str]:
        """
        Get a new innovation number for a node.
        The optional nodes indicate, between which nodes the new node is placed
        :param node1: one of nodes, between the new node is placed
        :param node2: the other node, between the new node is placed
        :return: an innovation number either as string or int
        """
        pass

    def get_connection_innovation_number(self, input_node: Node = None, output_node: Node = None) -> Union[int, str]:
        """
        Get a new innovation number for a connection.
        The optional nodes indicate, between which nodes the new connection is placed
        :param input_node: the input node for the connection
        :param output_node:  the output node for the connection
        :return: an innovation number either as string or int
        """
        pass
