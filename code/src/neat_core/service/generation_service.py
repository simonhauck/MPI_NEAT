from typing import Callable

import numpy as np

import neat_core.service.reproduction_service as rp
from neat_core.models.agent import Agent
from neat_core.models.connection import Connection
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.models.node import Node, NodeType
from neat_core.models.species import Species
from neat_core.optimizer.neat_config import NeatConfig


def create_initial_generation(amount_input_nodes: int, amount_output_nodes: int,
                              activation_function: Callable[[float], float],
                              generator: InnovationNumberGeneratorInterface,
                              config: NeatConfig, seed: int = None) -> Generation:
    """
    Create an initial generation, with the specified genome information. The network will be fully connected.
    :param amount_input_nodes: the amount if input nodes in the neural network
    :param amount_output_nodes: the amount of output nodes in the neural network
    :param activation_function: the used activation function for the nodes
    :param generator: implementation to generate innovation numbers
    :param config: a NeatConfig to set weights and population size
    :param seed: fo generate deterministic random values
    :return: the generated generation
    """
    rnd = np.random.RandomState(seed)

    # Create all genomes
    initial_genome = create_initial_genome(amount_input_nodes, amount_output_nodes, activation_function, rnd, config,
                                           generator)

    return _build_generation_from_genome(initial_genome, rnd, config)


def create_initial_generation_genome(genome: Genome, generator: InnovationNumberGeneratorInterface, config: NeatConfig,
                                     seed=None) -> Generation:
    rnd = np.random.RandomState(seed)

    # The naming of the stored genome, doest not necessary be conform with the innovationNumberGenerator.
    # So create new genome with the same connection structure but with generated innovation numbers
    tmp_node_key = {}
    new_nodes = []
    for node in genome.nodes:
        new_node = rp.deep_copy_node(node)
        new_node.innovation_number = generator.get_node_innovation_number()

        # Store node
        new_nodes.append(new_node)
        tmp_node_key[node.innovation_number] = new_node.innovation_number

    new_connections = []
    for connection in genome.connections:
        new_connection = rp.deep_copy_connection(connection)
        new_connection.innovation_number = generator.get_connection_innovation_number()
        # Match old numbers to newly generated innovation numbers
        new_connection.input_node = tmp_node_key[connection.input_node]
        new_connection.output_node = tmp_node_key[connection.output_node]

        # Store connection
        new_connections.append(new_connection)

    initial_genome = rp.deep_copy_genome(genome)
    initial_genome.connections = new_connections
    initial_genome.nodes = new_nodes

    return _build_generation_from_genome(initial_genome, rnd, config)


def _build_generation_from_genome(initial_genome: Genome, rnd: np.random.RandomState, config: NeatConfig) -> Generation:
    # Deep copy genome and set new weights
    genomes = [rp.set_new_genome_weights(rp.deep_copy_genome(initial_genome), seed=rnd.randint(2 ** 24), config=config)
               for _ in range(config.population_size)]

    agents = [Agent(genome) for genome in genomes]
    species = Species(representative=genomes[0], members=agents)

    return Generation(0, agents, [species])


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