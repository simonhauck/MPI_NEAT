from typing import Callable, List

import numpy as np

import neat_core.service.reproduction_service as rp
from neat_core.models.agent import Agent
from neat_core.models.connection import Connection
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.models.species import Species
from neat_core.optimizer.generator.agent_id_generator_interface import AgentIDGeneratorInterface
from neat_core.optimizer.generator.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.optimizer.generator.species_id_generator_interface import SpeciesIDGeneratorInterface
from neat_core.optimizer.neat_config import NeatConfig
from utils.fitness_evaluation import fitness_evaluation_utils


def create_initial_generation(amount_input_nodes: int, amount_output_nodes: int,
                              activation_function: Callable[[float], float],
                              inno_generator: InnovationNumberGeneratorInterface,
                              species_id_generator: SpeciesIDGeneratorInterface,
                              agent_id_generator: AgentIDGeneratorInterface,
                              config: NeatConfig, seed: int) -> Generation:
    """
    Create an initial generation, with the specified genome information. The network will be fully connected.
    :param amount_input_nodes: the amount if input nodes in the neural network
    :param amount_output_nodes: the amount of output nodes in the neural network
    :param activation_function: the used activation function for the nodes
    :param inno_generator: implementation to generate innovation numbers
    :param species_id_generator a generator to get species ids
    :param agent_id_generator a generator to get ids for agents
    :param config: a NeatConfig to set weights and population size
    :param seed: to generate deterministic random values
    :return: the generated generation
    """
    rnd = np.random.RandomState(seed)

    # Create genome structure
    genome_structure = create_genome_structure(amount_input_nodes, amount_output_nodes, activation_function, config,
                                               inno_generator)

    return _build_generation_from_genome(genome_structure, species_id_generator, agent_id_generator, seed, rnd, config)


def create_initial_generation_genome(genome: Genome, inno_generator: InnovationNumberGeneratorInterface,
                                     species_id_generator: SpeciesIDGeneratorInterface,
                                     agent_id_generator: AgentIDGeneratorInterface, config: NeatConfig,
                                     seed: int) -> Generation:
    """
    Create an initial generation, with the given genome. This function only uses the structure of the genome. The nodes
    and connections will receive new innovation numbers according to the given InnovationNumberGeneratorInterface.
    :param genome: the initial genome, that will be copied
    :param inno_generator: to get the innovation numbers
    :param species_id_generator a generator to get species ids
    :param agent_id_generator a generator to get ids for agents
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
        new_node.innovation_number = inno_generator.get_node_innovation_number()

        # Store node
        new_nodes.append(new_node)
        tmp_node_key[node.innovation_number] = new_node.innovation_number

    new_connections = []
    for connection in genome.connections:
        new_connection = rp.deep_copy_connection(connection)
        new_connection.innovation_number = inno_generator.get_connection_innovation_number()
        # Match old numbers to newly generated innovation numbers
        new_connection.input_node = tmp_node_key[connection.input_node]
        new_connection.output_node = tmp_node_key[connection.output_node]

        # Store connection
        new_connections.append(new_connection)

    initial_genome = rp.deep_copy_genome(genome)
    initial_genome.connections = new_connections
    initial_genome.nodes = new_nodes

    rnd = np.random.RandomState(seed)

    return _build_generation_from_genome(initial_genome, species_id_generator, agent_id_generator, seed, rnd, config)


def _randomize_weight_bias(genome: Genome, rnd: np.random.RandomState(), config: NeatConfig) -> Genome:
    genome = rp.set_new_genome_bias(genome, rnd, config)
    genome = rp.set_new_genome_weights(genome, rnd, config)
    return genome


def _build_generation_from_genome(initial_genome: Genome, species_id_generator: SpeciesIDGeneratorInterface,
                                  agent_id_generator: AgentIDGeneratorInterface, generation_seed: int,
                                  rnd: np.random.RandomState,
                                  config: NeatConfig) -> Generation:
    """
    Build a generation from the given genome. The genome will be copied and the weight and biases will be randomized
    :param initial_genome: the genome as initial structure
    :param species_id_generator a generator to get species ids
    :param a generator to get agent ids
    :param generation_seed the seed for the generation
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

    agents = [Agent(agent_id_generator.get_agent_id(), genome) for genome in genomes]
    species = Species(id_=species_id_generator.get_species_id(), representative=genomes[0], members=agents)

    return Generation(0, generation_seed, agents, [species])


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

    return Genome(seed=None, nodes=input_nodes + output_nodes, connections=connections)


def get_best_genomes_from_species(species_list: List[Species], min_species_size) -> List[Genome]:
    """
    Get the best genomes from each species with more members then the given min_species_size. The agent, with the best
    fitness value will be included, even if the species has less members then the given min_species_size
    :param species_list: a list of species, from which the best members should be extraced
    :param min_species_size: the min size if each species, that the best agent will be copied
    :return: a list with the best genomes
    """
    best_generation_agent = None
    best_genomes_list = []

    for species in species_list:
        best_species_agent = fitness_evaluation_utils.get_best_agent(species.members)

        # Check if the species champions is the generation champion
        if best_generation_agent is None or best_generation_agent.fitness < best_species_agent.fitness:
            best_generation_agent = best_species_agent

        # Check if species agent should be added
        if len(species.members) >= min_species_size:
            best_genomes_list.append(best_species_agent.genome)

    # Check if best generation agent is in best genomes list
    if best_generation_agent is not None and best_generation_agent.genome not in best_genomes_list:
        best_genomes_list.append(best_generation_agent.genome)

    return best_genomes_list
