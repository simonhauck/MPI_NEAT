from neat_core.models.genome import Genome
from neat_core.optimizer.neat_config import NeatConfig


# TODO Test
def calculate_genetic_distance(genome1: Genome, genome2: Genome, config: NeatConfig) -> float:
    return _calculate_genetic_distance_nodes(genome1, genome2, config) + _calculate_genetic_distance_connections(
        genome1, genome2, config)


def _calculate_genetic_distance_nodes(genome1: Genome, genome2: Genome, config: NeatConfig) -> float:
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
