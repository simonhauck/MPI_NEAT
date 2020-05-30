import math
from typing import List, Tuple

import numpy as np

from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.species import Species
from neat_core.optimizer.generator.agent_id_generator_interface import AgentIDGeneratorInterface
from neat_core.optimizer.generator.species_id_generator_interface import SpeciesIDGeneratorInterface
from neat_core.optimizer.neat_config import NeatConfig
from utils.fitness_evaluation import fitness_evaluation_utils


def calculate_genetic_distance(genome1: Genome, genome2: Genome, config: NeatConfig) -> float:
    """
    Calculate the compatibility between the two given genomes. This includes the nodes and connections
    :param genome1: the first genome
    :param genome2: the second genome
    :param config: the config that specifies the compatibility functions
    :return: the compatibility value between the two nodes
    """
    return _calculate_genetic_distance_nodes(genome1, genome2, config) + _calculate_genetic_distance_connections(
        genome1, genome2, config)


def _calculate_genetic_distance_nodes(genome1: Genome, genome2: Genome, config: NeatConfig) -> float:
    """
    Calculate the genetic distance between the nodes of the given genomes
    :param genome1: the first genome
    :param genome2: the second genome
    :param config: that specifies the compatibility factors
    :return: the compatibility factor between the nodes
    """
    g1_nodes_innovation_numbers = set(node.innovation_number for node in genome1.nodes)
    g2_nodes_innovation_numbers = set(node.innovation_number for node in genome2.nodes)

    # Calculate matching & disjoint node genes
    matching_genes = g1_nodes_innovation_numbers & g2_nodes_innovation_numbers
    g1_disjoint_genes = g1_nodes_innovation_numbers - matching_genes
    g2_disjoint_genes = g2_nodes_innovation_numbers - matching_genes

    # Calculate disjoint node genes value
    amount_disjoint_genes = (len(g1_disjoint_genes) + len(g2_disjoint_genes))
    max_genome_size = (max(len(genome1.nodes), len(genome2.nodes)))
    disjoint_genes_result = (config.compatibility_factor_disjoint_genes * amount_disjoint_genes) / max_genome_size

    # Calculate matching node genes
    # Create dictionary for matching genes
    g1_node_dict = {node.innovation_number: node for node in
                    (filter(lambda node: node.innovation_number in matching_genes, genome1.nodes))}
    matching_genes_difference_sum = 0
    # TODO maybe check also activation function
    for g2_node in genome2.nodes:
        if g2_node.innovation_number in matching_genes:
            # Matching node
            matching_genes_difference_sum += abs(g2_node.bias - g1_node_dict[g2_node.innovation_number].bias)
    matching_genes_result = matching_genes_difference_sum / len(
        matching_genes) * config.compatibility_factor_matching_genes

    return disjoint_genes_result + matching_genes_result


def _calculate_genetic_distance_connections(genome1: Genome, genome2: Genome, config: NeatConfig) -> float:
    """
    Calculate the genetic distance with the connections of the given genomes.
    :param genome1: the first genome
    :param genome2: the second genome
    :param config: that specifies the compatibility factors
    :return: the genetic distance for the connections
    """
    g1_connection_innovation_numbers = set(con.innovation_number for con in genome1.connections)
    g2_connection_innovation_numbers = set(con.innovation_number for con in genome2.connections)

    # Calculate matching & disjoint connection genes
    matching_genes = g1_connection_innovation_numbers & g2_connection_innovation_numbers
    g1_disjoint_genes = g1_connection_innovation_numbers - matching_genes
    g2_disjoint_genes = g2_connection_innovation_numbers - matching_genes

    # Calculate disjoint connection genes value
    amount_disjoint_genes = (len(g1_disjoint_genes) + len(g2_disjoint_genes))
    max_genome_size = (max(len(genome1.connections), len(genome2.connections)))
    disjoint_genes_result = (config.compatibility_factor_disjoint_genes * amount_disjoint_genes) / max_genome_size

    # Calculate matching connection genes
    # Create dictionary for matching genes
    g1_con_dict = {con.innovation_number: con for con in
                   (filter(lambda con: con.innovation_number in matching_genes, genome1.connections))}
    matching_genes_difference_sum = 0
    # TODO maybe check also if gene is enabled or disabled
    for g2_con in genome2.connections:
        if g2_con.innovation_number in matching_genes:
            # Matching node
            matching_genes_difference_sum += abs(g2_con.weight - g1_con_dict[g2_con.innovation_number].weight)
    matching_genes_result = matching_genes_difference_sum / len(
        matching_genes) * config.compatibility_factor_matching_genes

    return disjoint_genes_result + matching_genes_result


def sort_agents_into_species(existing_species: List[Species], agents: List[Agent],
                             species_id_generator: SpeciesIDGeneratorInterface, config: NeatConfig) -> List[Species]:
    """
    Sort the given agents into the given list of species, according to the compatibility. If no matching species is
    found for an agent, a new species is created and the agent is placed inside it.
    Note: The existing members of a species are not deleted!
    :param existing_species: a list of existing species.
    :param agents: a list of agents that should be placed into species
    :param species_id_generator to generate new ids for species
    :param config: a neat config with the required compatibility parameters
    :return: the list of species with the sorted agents
    """
    for agent in agents:
        for species in existing_species:
            compatibility = calculate_genetic_distance(agent.genome, species.representative, config)
            if compatibility <= config.compatibility_threshold:
                species.members.append(agent)
                break
        else:
            # Not found a matching element,
            new_species_id = species_id_generator.get_species_id()
            new_species = Species(new_species_id, agent.genome, [agent])
            existing_species.append(new_species)

    return existing_species


def update_fitness_species(generation: Generation) -> Generation:
    """
    Update the max fitness and the corresponding generation of each species with the values from its members.
    :param generation: which contains the species
    :return: the updated generation
    """
    for species in generation.species_list:
        best_fitness_current_generation = fitness_evaluation_utils.get_best_agent(species.members).fitness

        # If fitness is none or lower, update the value
        if species.max_species_fitness is None or species.max_species_fitness < best_fitness_current_generation:
            species.max_species_fitness = best_fitness_current_generation
            species.generation_max_species_fitness = generation.number

    return generation


def get_allowed_species_for_reproduction(generation: Generation, max_stagnant_generations: int) -> List[Species]:
    """
    Get the allowed species of the generation that are allow to reproduce
    :param generation: the last generation with the generation number and species
    :param max_stagnant_generations: how long the fitness of a species can stagnate, before it can't reproduce
    :return: the list with species
    """
    allowed_species = []
    for species in generation.species_list:
        # Species, which have improved the fitness x generations are allowed
        if species.generation_max_species_fitness + max_stagnant_generations >= generation.number:
            allowed_species.append(species)
    return allowed_species


def calculate_adjusted_fitness(species_list: List[Species], min_fitness: float, max_fitness: float) -> List[Species]:
    """
    Calculate the adjusted fitness for each species
    :param species_list: a list of species, for which the adjusted fitness should be calculated
    :param min_fitness: the minimum fitness of the generation
    :param max_fitness: the maximum fitness of the generation
    :return: the updated species
    """
    fitness_range = max(1.0, max_fitness - min_fitness)
    for species in species_list:
        mean_species_fitness = np.mean([member.fitness for member in species.members])
        species.adjusted_fitness = (mean_species_fitness - min_fitness) / fitness_range

    return species_list


def calculate_amount_offspring(species_list: List[Species], amount_offspring) -> List[int]:
    """
    Calculate the amount of offspring for each species.
    :param species_list: the list fo species. Can not be None or empty!
    :param amount_offspring: the combined amount of offspring
    :return: a list with integer values, corresponding to the species
    """
    assert species_list is not None
    assert len(species_list) != 0

    off_spring_list = []
    sum_adjusted_fitness = sum([s.adjusted_fitness for s in species_list])
    remaining_offspring = amount_offspring

    # Calculate initial distribution
    for species in species_list:
        off_spring_species = (species.adjusted_fitness / sum_adjusted_fitness) * amount_offspring
        # The remaining species will be distributed later
        off_spring_species = math.floor(off_spring_species)
        # Reduce remaining off spring and add value to list
        remaining_offspring -= off_spring_species
        off_spring_list.append(off_spring_species)

    # Assign remaining offspring
    for i in range(remaining_offspring):
        off_spring_list[i] = off_spring_list[i] + 1

    return off_spring_list


def remove_low_genomes(species_list: List[Species], remove_percentage: float) -> List[Species]:
    """
    Remove the given percentage of low genomes from every species
    :param species_list: the list of species, which members should be modified
    :param remove_percentage: the percentage of low genomes that should be removed e.g 0.2 means the lower 20% percent
    :return: the updated species list
    """
    for species in species_list:
        remove_agents = math.floor(remove_percentage * len(species.members))
        sorted_members = sorted(species.members, key=lambda member: member.fitness)
        species.members = sorted_members[remove_agents:]

    return species_list


def create_offspring_pairs(species: Species, amount_offspring: int,
                           agent_id_generator: AgentIDGeneratorInterface,
                           generation: Generation,
                           rnd: np.random.RandomState,
                           config: NeatConfig) -> List[Tuple[int, int, int]]:
    """
    Create tuples, with ids of agents, that should be used in the crossover.
    :param species: the species, for which the crossover values should be generated
    :param amount_offspring: the amount of offspring for the given species
    :param agent_id_generator: the id generator for the agents
    :param generation the generation with all its members
    :param rnd: the random generator, to select the parents
    :param config: the neat config
    :return: a list with tuples. The tuple contains the id of the first parent, the second parent and the child id
    """
    assert len(species.members) != 0

    result_list = []
    species_len = len(species.members)
    for i in range(amount_offspring):
        # TODO add probability only mutation
        # TODO add reproduction with different species
        first_parent_index = rnd.randint(species_len)
        second_parent_index = rnd.randint(species_len)
        new_agent_id = agent_id_generator.get_agent_id()

        first_parent_id = species.members[first_parent_index].id
        second_parent_id = species.members[second_parent_index].id

        result_list.append((first_parent_id, second_parent_id, new_agent_id))

    return result_list


def select_new_representative(species: Species, rnd: np.random.RandomState) -> Species:
    """
    Assign a randomly selected genome from its members as new representative
    :param species: the species, with its members
    :param rnd: the random generator to select the representative
    :return: the updated species
    """
    assert len(species.members) != 0
    representative_index = rnd.randint(len(species.members))
    species.representative = species.members[representative_index].genome
    return species


def reset_species(species: Species) -> Species:
    """
    Reset the species members and the adjusted fitness value
    :param species: the species, that should be reset
    :return: the updated species
    """
    species.members = []
    species.adjusted_fitness = None
    return species


def get_species_with_members(species_list: List[Species]) -> List[Species]:
    """
    Get the species, which contain at least one member
    :param species_list: the original species list
    :return: a list with species, that contain more than one member
    """
    return list(filter(lambda species: len(species.members) >= 1, species_list))
