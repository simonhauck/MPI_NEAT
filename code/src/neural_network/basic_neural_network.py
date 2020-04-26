from typing import List, Dict, Union

import numpy as np
from loguru import logger

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neural_network.neural_network_interface import NeuralNetworkInterface


class BasicNeuron(object):

    def __init__(self, weights: np.ndarray, input_keys: np.ndarray) -> None:
        assert len(weights) == len(input_keys)
        self.val = 0.0
        self.weights = weights
        self.input_keys = input_keys
        self.inputs: np.ndarray = np.zeros([len(input_keys)], dtype=float)


class BasicNeuralNetwork(NeuralNetworkInterface):

    def __init__(self):
        self.input_neurons: Dict[Union[int, str]][BasicNeuron] = {}
        self.output_neurons: Dict[Union[int, str]][BasicNeuron] = {}
        self.all_neurons: Dict[Union[int, str]][BasicNeuron] = {}

        # Innovation number of nodes to determine calculation order
        self.order: List[Union[int, str]] = []

    def build(self, genome: Genome) -> None:
        logger.trace("Building neural network with genome {}".format(genome.id))
        # Set input neurons

    def reset(self) -> None:
        super().reset()

    def activate(self, inputs: List[float]) -> List[float]:
        return super().activate(inputs)

    @staticmethod
    def __sort_connections(connections: List[Connection]) -> Dict[Union[int, str], List[Connection]]:
        """
        Sort the given connections according to the output node.
        :param connections: all connections that should be sorted. Can be empty.
        :return: a dictionary. The key is the innovation number of the node, the value is a list with connections
        """
        result = {}
        for connection in connections:
            if connection.output_node not in result:
                result[connection.output_node] = [connection]
            else:
                result[connection.output_node].append(connection)
        return result
