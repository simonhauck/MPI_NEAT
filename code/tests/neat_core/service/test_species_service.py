from unittest import TestCase

import neat_core.service.species_service as ss
from neat_core.activation_function import step_function
from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.neat_config import NeatConfig


class SpeciesServiceTest(TestCase):

    def setUp(self) -> None:
        self.g1_nodes = [
            Node(1, NodeType.INPUT, 0, step_function, 0),
            Node(2, NodeType.INPUT, 0, step_function, 0),
            Node(3, NodeType.OUTPUT, 1.2, step_function, 1),
            Node(4, NodeType.HIDDEN, 1.5, step_function, 0.5),
            Node(6, NodeType.HIDDEN, 0.5, step_function, 0.5),
            Node(7, NodeType.HIDDEN, 0.2, step_function, 0.25)
        ]
        self.g1_connections = [
            Connection(1, 1, 3, 1.2, True),
            Connection(2, 2, 3, 0.5, False),
            Connection(3, 1, 4, -1.2, True),
            Connection(4, 4, 3, 0.2, True),
            Connection(5, 2, 6, 2.0, True),
            Connection(6, 6, 3, -1.1, False)
        ]
        self.g1 = Genome(1, 1, self.g1_nodes, self.g1_connections)

        self.g2_nodes = [
            Node(1, NodeType.INPUT, 0, step_function, 0),
            Node(2, NodeType.INPUT, 0, step_function, 0),
            Node(3, NodeType.OUTPUT, 0.2, step_function, 1),
            Node(4, NodeType.HIDDEN, 1.2, step_function, 0.5),
            Node(5, NodeType.HIDDEN, 2.8, step_function, 0.5)
        ]

        self.g2_connections = [
            Connection(1, 1, 3, 0.8, True),
            Connection(2, 2, 3, 1.5, True),
            Connection(3, 1, 4, 1.2, True),
            Connection(4, 4, 3, 3.2, True),
            Connection(6, 6, 3, -1.1, False),
            Connection(7, 6, 3, -0.1, False),
            Connection(8, 1, 4, -1.1, False)
        ]
        self.g2 = Genome(2, 2, self.g2_nodes, self.g2_connections)

        self.config = NeatConfig(compatibility_factor_matching_genes=1, compatibility_factor_disjoint_genes=2)

    def test_calculate_genetic_distance(self):
        genetic_distance = ss.calculate_genetic_distance(self.g1, self.g2, self.config)
        genetic_distance2 = ss.calculate_genetic_distance(self.g2, self.g1, self.config)

        disjoint_gene_value_node = (3 / 6) * self.config.compatibility_factor_disjoint_genes
        matching_genes_value_node = 0.325 * self.config.compatibility_factor_matching_genes
        disjoint_gene_value_connection = (3 / 7) * self.config.compatibility_factor_disjoint_genes
        matching_genes_value_connection = 1.36 * self.config.compatibility_factor_matching_genes

        self.assertEqual(genetic_distance, genetic_distance2)
        self.assertEqual(
            disjoint_gene_value_connection + matching_genes_value_connection + disjoint_gene_value_node +
            matching_genes_value_node, genetic_distance)

    def test_calculate_genetic_distance_nodes(self):
        disjoint_gene_value = (3 / 6) * self.config.compatibility_factor_disjoint_genes
        matching_genes_value = 0.325 * self.config.compatibility_factor_matching_genes
        self.assertEqual(disjoint_gene_value + matching_genes_value,
                         ss._calculate_genetic_distance_nodes(self.g1, self.g2, self.config))

    def test_calculate_genetic_distance_connections(self):
        disjoint_gene_value = (3 / 7) * self.config.compatibility_factor_disjoint_genes
        matching_genes_value = 1.36 * self.config.compatibility_factor_matching_genes
        self.assertEqual(disjoint_gene_value + matching_genes_value,
                         ss._calculate_genetic_distance_connections(self.g1, self.g2, self.config))
