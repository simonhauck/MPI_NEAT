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
    genome_structure = create_genome_structure(amount_input_nodes, amount_output_nodes, activation_function, config,
                                               generator)

    return _build_generation_from_genome(genome_structure, rnd, config)


def create_initial_generation_genome(genome: Genome, generator: InnovationNumberGeneratorInterface, config: NeatConfig,
                                     seed=None) -> Generation:
    """
    Create an initial generation, with the given genome. This function only uses the structure of the genome. The nodes
    and connections will receive new innovation numbers according to the given InnovationNumberGeneratorInterface.
    :param genome: the initial genome, that will be copied
    :param generator: to get the innovation numbers
    :param config: that contains min, max weight and population size
    :param seed: for deterministic weights
    :return: a new initialized generation, with number 0, the corresponding agents and one species
    """

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

    rnd = np.random.RandomState(seed)

    return _build_generation_from_genome(initial_genome, rnd, config)


def _randomize_weight_bias(genome: Genome, rnd: np.random.RandomState(), config: NeatConfig) -> Genome:
    genome = rp.set_new_genome_bias(genome, rnd, config)
    genome = rp.set_new_genome_weights(genome, rnd, config)
    return genome


def _build_generation_from_genome(initial_genome: Genome, rnd: np.random.RandomState, config: NeatConfig) -> Generation:
    """
    Build a generation from the given genome. The genome will be copied and the weight and biases will be randomized
    :param initial_genome: the genome as initial structure
    :param rnd: a random generator to generate the seeds
    :param config: the neat config to specify the weights bounds
    :return: a new initialized generation with number 0, the created agents, and one species
    """
    # Deep copy genome and set new weights
    genomes = []
    for _ in range(config.population_size):
        seed = rnd.randint(2 ** 24)
        rnd_generator_genome = np.random.RandomState(seed)

        # Copy genome, set new values and save seed
        copied_genome = rp.deep_copy_genome(initial_genome)
        copied_genome = _randomize_weight_bias(copied_genome, rnd_generator_genome, config)
        copied_genome.seed = seed
        genomes.append(copied_genome)

    agents = [Agent(genome) for genome in genomes]
    species = Species(representative=genomes[0], members=agents)

    return Generation(0, agents, [species])


def create_genome_structure(amount_input_nodes: int, amount_output_nodes: int, activation_function,
                            config: NeatConfig, generator: InnovationNumberGeneratorInterface) -> Genome:
    """
    Create an initial genome struture with the given amount of input and output nodes. The nodes will be fully
    connected,     that means, that every input node will be connected to every output node. The bias of the nodes, as
    well as the connections will have the value 0! They must be set before usage
    :param amount_input_nodes: the amount of input nodes, that will be placed in the genome
    :param amount_output_nodes: the amount of output nodes, that will be placed in the genome
    :param activation_function: the activation function for the nodes
    :param config: the config
    :param generator: for the innovation numbers for nodes and connections
    :return: the generated genome
    """
    input_nodes = [
        Node(generator.get_node_innovation_number(), NodeType.INPUT, bias=0,
             activation_function=activation_function, x_position=0.0)
        for _ in range(amount_input_nodes)]

    output_nodes = [Node(generator.get_node_innovation_number(), NodeType.OUTPUT, bias=0,
                         activation_function=activation_function, x_position=1.0)
                    for _ in range(amount_output_nodes)]

    connections = []
    for input_node in input_nodes:
        for output_node in output_nodes:
            connections.append(Connection(
                innovation_number=generator.get_connection_innovation_number(),
                input_node=input_node.innovation_number,
                output_node=output_node.innovation_number,
                weight=0,
                enabled=True))

    return Genome(id_=0, seed=None, nodes=input_nodes + output_nodes, connections=connections)
