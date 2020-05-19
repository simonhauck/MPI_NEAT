from typing import List

from neat_core.models.agent import Agent
from neat_core.models.genome import Genome
from neat_core.models.species import Species
from neat_core.models.species_id_generator import SpeciesIDGeneratorInterface
from neat_core.optimizer.neat_config import NeatConfig


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
