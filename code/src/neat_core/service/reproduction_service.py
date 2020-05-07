import numpy as np

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node
from neat_core.optimizer.neat_config import NeatConfig


def deep_copy_genome(genome: Genome) -> Genome:
    """
    Make a deep copy of the given genome and return the new instance
    :param genome: the genome to be copied
    :return: the newly created instance
    """
    nodes = [deep_copy_node(original_node) for original_node in genome.nodes]
    connections = [deep_copy_connection(connection) for connection in genome.connections]

    return Genome(genome.id, genome.seed, nodes, connections)


def deep_copy_node(node: Node) -> Node:
    """
    Make a deep copy of the given node and return the new instance
    :param node: the node to be copied
    :return: the newly created node
    """
    return Node(node.innovation_number, node.node_type, node.activation_function, node.x_position)


def deep_copy_connection(connection: Connection) -> Connection:
    """
    Make a deep copy of the given connection and return the new instance
    :param connection: the connection to be copied
    :return: the newly created connection
    """
    return Connection(connection.innovation_number, connection.input_node, connection.output_node, connection.weight,
                      connection.enabled)


def set_new_genome_weights(genome: Genome, seed: int, config: NeatConfig) -> Genome:
    """
    Set new weights for the connections and bias in the genome. The given seed will be set as seed for the genome.
    :param genome: the genome, which weights should be randomized
    :param seed: the seed that should be used
    :param config: the neat config that specifies max and min weight
    :return: the modified genome
    """
    rnd = np.random.RandomState(seed)
    genome.seed = seed

    for connection in genome.connections:
        connection.weight = rnd.uniform(low=config.connection_min_weight, high=config.connection_max_weight)

    return genome
