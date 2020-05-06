from typing import List, Callable

import numpy as np

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.neat_config import NeatConfig


def create_initial_genomes(input_nodes: int, output_nodes: int,
                           activation_function: Callable[[float], float],
                           generator: InnovationNumberGeneratorInterface,
                           config: NeatConfig, seed: int = None) -> List[Genome]:
    genome_list = []
    rnd = np.random.RandomState()
    rnd.seed(seed)
    return []


def create_initial_genome(amount_input_nodes: int, amount_output_nodes: int, activation_function,
                          rnd: np.random.RandomState, config: NeatConfig,
                          generator: InnovationNumberGeneratorInterface) -> Genome:
    """
    Create an initial genome with the given amount of input and output nodes. The nodes will be fully connected,
    that means, that every input node will be connected to every output node.
    :param amount_input_nodes: the amount of input nodes, that will be placed in the genome
    :param amount_output_nodes: the amount of output nodes, that will be placed in the genome
    :param activation_function: the activation function for the nodes
    :param rnd: a random generator, to generate a seed for the genome
    :param config: the config which specifies the min and max weight
    :param generator: for the innovation numbers for nodes and connections
    :return: the generated genome
    """
    input_nodes = [Node(generator.get_node_innovation_number(), NodeType.INPUT, activation_function, 0.0)
                   for _ in range(amount_input_nodes)]

    output_nodes = [Node(generator.get_node_innovation_number(), NodeType.OUTPUT, activation_function, 1.0)
                    for _ in range(amount_output_nodes)]

    seed_genome = rnd.randint(2 ** 24)
    genome_rnd = np.random.RandomState(seed=seed_genome)
    connections = []

    for input_node in input_nodes:
        for output_node in output_nodes:
            weight = genome_rnd.uniform(low=config.connection_min_weight, high=config.connection_max_weight)
            connections.append(Connection(
                innovation_number=generator.get_connection_innovation_number(),
                input_node=input_node.innovation_number,
                output_node=output_node.innovation_number,
                weight=weight,
                enabled=True))

    return Genome(id_=0, seed=seed_genome, nodes=input_nodes + output_nodes, connections=connections)
