from typing import List

import numpy as np

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.generator.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.optimizer.neat_config import NeatConfig


def deep_copy_genome(genome: Genome) -> Genome:
    """
    Make a deep copy of the given genome and return the new instance
    :param genome: the genome to be copied
    :return: the newly created instance
    """
    nodes = [deep_copy_node(original_node) for original_node in genome.nodes]
    connections = [deep_copy_connection(connection) for connection in genome.connections]

    return Genome(genome.seed, nodes, connections)


def deep_copy_node(node: Node) -> Node:
    """
    Make a deep copy of the given node and return the new instance
    :param node: the node to be copied
    :return: the newly created node
    """
    return Node(node.innovation_number, node.node_type, node.bias, node.activation_function, node.x_position)


def deep_copy_connection(connection: Connection) -> Connection:
    """
    Make a deep copy of the given connection and return the new instance
    :param connection: the connection to be copied
    :return: the newly created connection
    """
    return Connection(connection.innovation_number, connection.input_node, connection.output_node, connection.weight,
                      connection.enabled)


def set_new_genome_weights(genome: Genome, rnd: np.random.RandomState, config: NeatConfig) -> Genome:
    """
    Set new weights for the connections and the genome with the given random generator.
    :param genome: the genome, which weights should be randomized
    :param rnd: random generator to receive the new weights
    :param config: the neat config that specifies max and min weight
    :return: the modified genome
    """
    for connection in genome.connections:
        connection.weight = rnd.uniform(low=config.connection_initial_min_weight,
                                        high=config.connection_initial_max_weight)

    return genome


def set_new_genome_bias(genome: Genome, rnd: np.random.RandomState, config: NeatConfig) -> Genome:
    """
    Set a new bias values in all nodes (except Input nodes) with the given random generator
    :param genome: the genome, where the bias value should be modified
    :param rnd: the random generator
    :param config: a neat config that specifies min and max values
    :return: the modified genome
    """
    for node in genome.nodes:
        if node.node_type == NodeType.INPUT:
            continue
        node.bias = rnd.uniform(low=config.bias_initial_min, high=config.bias_initial_max)
    return genome


def mutate_bias(genome: Genome, rnd: np.random.RandomState, config: NeatConfig) -> Genome:
    """
    Mutate the bias of the nodes
    :param genome: the genome, with nodes
    :param rnd: a random generator to determine, which nodes should be mutated and how
    :param config: neat config that specifies the probabilities
    :return: the modified genome
    """
    for node in genome.nodes:
        # Input nodes do not have a bias
        if node.node_type == NodeType.INPUT:
            continue

        if rnd.uniform(0, 1) <= config.probability_bias_mutation:
            # Assign random bias or perturb value
            if rnd.uniform(0, 1) <= config.probability_random_bias_mutation:
                node.bias = rnd.uniform(low=config.bias_initial_min, high=config.bias_initial_max)
            else:
                # Check how the connection weight should be mutated
                mutation_type = config.bias_mutation_type
                if mutation_type == "uniform":
                    node.bias += rnd.uniform(low=-config.bias_mutation_uniform_max_change,
                                             high=config.bias_mutation_uniform_max_change)
                elif mutation_type == "normal":
                    node.bias += rnd.normal(loc=0, scale=config.bias_mutation_normal_sigma)
                else:
                    raise AssertionError("Unknown type of mutation type. Must be 'uniform' or 'normal'")

                node.bias = np.clip(node.bias, a_min=config.bias_min, a_max=config.bias_max)
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
                connection.weight = rnd.uniform(low=config.connection_initial_min_weight,
                                                high=config.connection_initial_max_weight)
            else:
                # Check how the connection weight should be mutated
                mutation_type = config.weight_mutation_type
                if mutation_type == "uniform":
                    connection.weight += rnd.uniform(-config.weight_mutation_uniform_max_change,
                                                     config.weight_mutation_uniform_max_change)
                elif mutation_type == "normal":
                    connection.weight += rnd.normal(loc=0, scale=config.weight_mutation_normal_sigma)
                else:
                    raise AssertionError("Unknown type of mutation type. Must be 'uniform' or 'normal'")

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
    # Check if connection should be mutated at all
    if rnd.uniform(0, 1) > config.probability_mutate_add_connection:
        return genome, None

    # Sort nodes
    sorted_nodes = genome.nodes.copy()
    sorted_nodes.sort(key=lambda node: node.x_position)

    # List with hidden and output nodes
    hidden_output_nodes = list(filter(lambda node: node.node_type != NodeType.INPUT, sorted_nodes))

    # If a selection is not possible, retry the specified amount of times
    for _ in range(config.mutate_connection_tries):
        in_node = sorted_nodes[rnd.randint(len(sorted_nodes))]

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

        out_node = possible_out_nodes[rnd.randint(len(possible_out_nodes))]

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
                                    weight=rnd.uniform(config.connection_initial_min_weight,
                                                       config.connection_initial_max_weight),
                                    enabled=True)

        genome.connections.append(new_connection)
        return genome, new_connection

    return genome, None


def mutate_add_node(genome: Genome, rnd: np.random.RandomState, generator: InnovationNumberGeneratorInterface,
                    config: NeatConfig) -> (Genome, Node, Connection, Connection):
    """
    Add with a given probability from the config a new node to the genome.
    A random connections is selected, which will be disabled. A new node will be placed between the in and out node of
    the connection. Then two new connections will be created, one which leads into the new node (weight=1) and one out
    (weight = weight of the disabled connection).
    :param genome: the genome that should be modified
    :param rnd: a random generator to determine if, the genome is mutated, and how
    :param generator: a generator for innovation number for nodes and connections
    :param config: a config that specifies the mutation params
    :return: the modified genome, as well as the generated node and the two connections (if they were mutated)
    """
    # Check if node should mutate
    if rnd.uniform(0, 1) > config.probability_mutate_add_node:
        return genome, None, None, None

    selected_connection = genome.connections[rnd.randint(0, len(genome.connections))]
    selected_connection.enabled = False

    in_node = next(x for x in genome.nodes if x.innovation_number == selected_connection.input_node)
    out_node = next(x for x in genome.nodes if x.innovation_number == selected_connection.output_node)

    # Select activation function either from one of the nodes
    new_node_activation = in_node.activation_function if rnd.uniform(0, 1) <= 0.5 else out_node.activation_function
    new_node_x_position = (in_node.x_position + out_node.x_position) / 2
    new_node = Node(
        generator.get_node_innovation_number(in_node, out_node),
        NodeType.HIDDEN,
        rnd.uniform(low=config.bias_initial_min, high=config.bias_initial_max),
        new_node_activation,
        new_node_x_position
    )

    new_connection_in = Connection(generator.get_connection_innovation_number(in_node, new_node),
                                   in_node.innovation_number, new_node.innovation_number, weight=1, enabled=True)
    new_connection_out = Connection(generator.get_connection_innovation_number(new_node, out_node),
                                    new_node.innovation_number, out_node.innovation_number,
                                    weight=selected_connection.weight, enabled=True)

    genome.nodes.append(new_node)
    genome.connections.append(new_connection_in)
    genome.connections.append(new_connection_out)

    return genome, new_node, new_connection_in, new_connection_out


def cross_over(more_fit_genome: Genome, less_fit_genome: Genome, rnd: np.random.RandomState, config: NeatConfig) \
        -> (List[Node], List[Connection]):
    """
    Perform the cross over between two parents. Matching genes (that are in both genomes) will be chosen randomly with
    the given rnd generator. Disjoint and access genes will be inherited from the more fit parent
    :param more_fit_genome: the genome with the higher fitness value
    :param less_fit_genome: the genome with the lower fitness value
    :param rnd: a random generator to determine which genes should be selected
    :param config: the config that specifies the reproduction behavior
    :return: a list with selected nodes and a list with selected connections
    """
    #  Create dictionary for nodes and connections of less fit genome
    less_fit_genome_nodes_dict = {node.innovation_number: node for node in less_fit_genome.nodes}
    less_fit_genome_connections_dict = {conn.innovation_number: conn for conn in less_fit_genome.connections}

    nodes_child = []
    # Iterate over nodes in more fit parent
    for more_fit_genome_node in more_fit_genome.nodes:
        # Check if node is a matching gene
        if more_fit_genome_node.innovation_number in less_fit_genome_nodes_dict:
            # Matching gene
            selected_node = more_fit_genome_node if rnd.uniform(0, 1) <= 0.5 else less_fit_genome_nodes_dict[
                more_fit_genome_node.innovation_number]
        else:
            # Disjoint or excess gene
            selected_node = more_fit_genome_node

        # Copy and add it to child
        copied_node = deep_copy_node(selected_node)
        nodes_child.append(copied_node)

    connections_child = []
    for more_fit_genome_connection in more_fit_genome.connections:

        # Check if node is a matching gene
        if more_fit_genome_connection.innovation_number in less_fit_genome_connections_dict:
            less_fit_genome_connection = less_fit_genome_connections_dict[more_fit_genome_connection.innovation_number]
            # Matching gene
            selected_connection = more_fit_genome_connection if rnd.uniform(0, 1) <= 0.5 else less_fit_genome_connection
            enable_connection = selected_connection.enabled

            # Check if gene should be re-enabled
            if not more_fit_genome_connection.enabled and not less_fit_genome_connection.enabled:
                enable_connection = rnd.uniform(0, 1) <= config.probability_enable_gene

        else:
            selected_connection = more_fit_genome_connection
            enable_connection = selected_connection.enabled

        copied_connection = deep_copy_connection(selected_connection)
        copied_connection.enabled = enable_connection

        connections_child.append(copied_connection)

    return nodes_child, connections_child
