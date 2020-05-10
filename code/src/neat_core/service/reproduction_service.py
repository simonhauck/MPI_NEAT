import numpy as np

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.models.node import Node, NodeType
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


def mutate_weights(genome: Genome, rnd: np.random.RandomState, config: NeatConfig) -> Genome:
    """
    Mutate the connection weights, using the given random generator and the config
    :param genome: the genome which weights should be mutated
    :param rnd: a random generator to determine, which weights and how much they will be changed
    :param config: a config that specifies the probability and magnitude of the changes
    :return: the mutated genome
    """
    for connection in genome.connections:
        # Should mutate weights?
        if rnd.uniform(0, 1) <= config.probability_weight_mutation:
            # Assign random weight or perturb existing weight?
            if rnd.uniform(0, 1) <= config.probability_random_weight_mutation:
                connection.weight = rnd.uniform(config.connection_min_weight, config.connection_max_weight)
            else:
                connection.weight += rnd.uniform(-config.weight_mutation_max_change, config.weight_mutation_max_change)
                connection.weight = np.clip(connection.weight, a_min=config.connection_min_weight,
                                            a_max=config.connection_max_weight)

    return genome


def mutate_add_connection(genome: Genome, rnd: np.random.RandomState, generator: InnovationNumberGeneratorInterface,
                          config: NeatConfig) -> (Genome, Connection):
    """
    Mutate the genome and add a new connection (if possible). The in and out node of the connection are chosen
    randomly with the given rnd generator, as well as the weight. If the in the config, recurrent networks are set to
    False, only feed forward connections can be made.
    :param genome: genome that should be modified
    :param rnd: the random generator
    :param generator: to generate innovation numbers
    :param config: the config that specifies, where and how the connection is created
    :return: the modified genome and the newly created connection. If adding a connection failed, it will be None
    """
    # Check if node should be mutated at all
    if rnd.uniform(0, 1) > config.probability_mutate_add_connection:
        return genome, None

    # Sort nodes
    sorted_nodes = genome.nodes.copy()
    sorted_nodes.sort(key=lambda node: node.x_position)

    # List with hidden and output nodes
    hidden_output_nodes = list(filter(lambda node: node.node_type != NodeType.INPUT, sorted_nodes))

    # If a selection is not possible, retry the specified amount of times
    for _ in range(config.mutate_connection_tries):
        in_node = sorted_nodes[rnd.randint(0, len(sorted_nodes))]

        if config.allow_recurrent:
            # In recurrent networks, all nodes except input nodes can be output nodes
            possible_out_nodes = hidden_output_nodes
        else:
            # Feed forward networks, require additional, that the x position of the in_node is smaller
            possible_out_nodes = filter(lambda node: in_node.x_position < node.x_position, hidden_output_nodes)

        # Convert from iterable to list
        possible_out_nodes = list(possible_out_nodes)
        if len(possible_out_nodes) == 0:
            continue

        out_node = hidden_output_nodes[rnd.randint(0, len(possible_out_nodes))]

        # Check if connection already exists
        exists = any(
            connection.input_node == in_node.innovation_number and
            connection.output_node == out_node.innovation_number
            for connection in genome.connections)
        # Double connections are not allowed
        if exists:
            continue

        innovation_number = generator.get_connection_innovation_number(in_node, out_node)
        new_connection = Connection(innovation_number, in_node.innovation_number, out_node.innovation_number,
                                    weight=rnd.uniform(config.connection_min_weight, config.connection_max_weight),
                                    enabled=True)

        genome.connections.append(new_connection)
        return genome, new_connection

    return genome, None
