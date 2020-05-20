from typing import List, Dict, Union, Callable

import numpy as np
from loguru import logger

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import NodeType
from neural_network.neural_network_interface import NeuralNetworkInterface


class BasicNeuron(object):

    def __init__(self, innovation_number: Union[int, str],
                 bias: float,
                 weights: np.ndarray,
                 input_keys: np.ndarray,
                 activation_function: Callable[[float], float],
                 x_position: float) -> None:
        assert len(weights) == len(input_keys)
        self.val: float = 0.0
        self.last_val: float = 0.0
        self.flag_calculated: bool = False
        self.innovation_number: Union[int, str] = innovation_number
        self.x_position: float = x_position

        self.bias: float = bias
        self.weights = np.ndarray = weights
        self.input_keys = np.ndarray = input_keys
        self.activation_function: Callable[[float], float] = activation_function


class BasicNeuralNetwork(NeuralNetworkInterface):

    def __init__(self):
        self.input_neurons: List[BasicNeuron] = []
        self.output_neurons: List[BasicNeuron] = []
        self.all_neurons: Dict[Union[int, str], BasicNeuron] = {}

        # Innovation number of nodes to determine calculation order
        self.order: List[Union[int, str]] = []

    def build(self, genome: Genome) -> None:
        """
        Build the neural network with the information encoded in the genome.
        :param genome: that encodes the neural network
        :return: None
        """
        logger.trace("Building neural network with genome {}".format(genome.id))

        # Sort neurons according to output
        sorted_connection = BasicNeuralNetwork._sort_connections(genome.connections)

        # Iterate over sorted nodes.
        for node in sorted(genome.nodes, key=lambda n: n.x_position):
            # Get keys and weights of active connections for each node
            connections_for_node = sorted_connection.get(node.innovation_number, [])
            input_keys = np.array([connection.input_node for connection in connections_for_node])
            weights = np.array([connection.weight for connection in connections_for_node])

            basic_neuron = BasicNeuron(node.innovation_number, node.bias, weights, input_keys, node.activation_function,
                                       node.x_position)

            # Add neuron to the dictionary for the calculations
            self.all_neurons[node.innovation_number] = basic_neuron

            if node.node_type == NodeType.INPUT:
                self.input_neurons.append(basic_neuron)
            else:
                # Hidden and Output neurons are added to the calculation order (input neurons mustn't be calculated)
                self.order.append(node.innovation_number)

                if node.node_type == NodeType.OUTPUT:
                    self.output_neurons.append(basic_neuron)

    def reset(self) -> None:
        """
        Reset the neural network to its initial state. All temporary stored values will be removed.
        :return: None
        """
        for node in self.all_neurons.values():
            node.flag_calculated = False
            node.val = 0
            node.last_val = 0

    def activate(self, inputs: List[float]) -> List[float]:
        """
        Activate the neural network with the given inputs
        :param inputs: a list of float input values. The size must match the size of input neurons
        :return: the result of the neural network. The size if the list matches the amount of output neurons
        """
        assert len(inputs) == len(self.input_neurons)

        # Set flags to false, store last value
        for neuron in self.all_neurons.values():
            neuron.flag_calculated = False

        # Set input neurons
        for input_neuron, input_value in zip(self.input_neurons, inputs):
            input_neuron.val = input_value
            input_neuron.flag_calculated = True

        # Iterate over neurons to be calculated
        for innovation_number in self.order:
            neuron = self.all_neurons[innovation_number]

            calculated_val = 0

            # Sum the weighted input values
            for i, input_key in zip(range(len(neuron.input_keys)), neuron.input_keys):
                calculated_val += neuron.weights[i] * self._get_input_value_from_neuron(input_key, neuron.x_position)

            # Add bias
            calculated_val += neuron.bias

            calculated_val = neuron.activation_function(calculated_val)
            logger.trace("Neuron Value - Key: {} -> Val: {}".format(neuron.innovation_number, calculated_val))

            # Set the flag and result
            neuron.flag_calculated = True
            neuron.last_val = neuron.val
            neuron.val = calculated_val

        # Get output neurons
        result = [output_neuron.val for output_neuron in self.output_neurons]
        logger.trace("Net activated: Output: {} | Input: {}".format(result, inputs))
        return result

    def _get_input_value_from_neuron(self, target_innovation_number: Union[int, str], x_position: float) -> float:
        """
        Returns the value of the given neuron. Depending on whether the connection is recurrent the target_neuron was
        already calculated the current or last value will be returned
        :param target_innovation_number: the innovation number of the target neuron
        :param x_position: the position of the neuron, which should be calculated
        :return: the value of the neuron
        """
        target_neuron = self.all_neurons[target_innovation_number]

        # Is connection recurrent? Then the value from the last run must be returned
        if x_position <= target_neuron.x_position:
            # Check if the neuron was already calculated. If the yes return the last_val. If not, the value stored in
            # val is the old value
            if target_neuron.flag_calculated:
                return target_neuron.last_val
            else:
                return target_neuron.val

        if not target_neuron.flag_calculated:
            raise ValueError(
                "The from x position of the target neuron is smaller, but is not yet calculated! Calculation order is "
                "broken Target xPosition {}, Given x Position: {}".format(
                    target_neuron.x_position, x_position))

        return target_neuron.val

    @staticmethod
    def _sort_connections(connections: List[Connection]) -> Dict[Union[int, str], List[Connection]]:
        """
        Sort the given connections according to the output node. Only the enabled connections will be considered
        :param connections: all connections that should be sorted. Can be empty.
        :return: a dictionary. The key is the innovation number of the node, the value is a list with connections
        """
        result = {}
        for connection in connections:
            # Check only active connections
            if connection.enabled is not True:
                continue

            # If key is not in list, add the value, else add the key to the list
            if connection.output_node not in result:
                result[connection.output_node] = [connection]
            else:
                result[connection.output_node].append(connection)
        return result
