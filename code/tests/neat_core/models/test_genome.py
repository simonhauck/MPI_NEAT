from unittest import TestCase

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType


class GenomeTest(TestCase):

    def test_genome(self):
        genome_empty = Genome(20)
        self.assertEqual(20, genome_empty.seed)
        self.assertEqual([], genome_empty.nodes)

        node_list = [
            Node(2, NodeType.INPUT, bias=0.5, activation_function=lambda x: x + 1, x_position=0),
            Node("node_number", NodeType.OUTPUT, bias=0.4, activation_function=lambda x: x + 1, x_position=1)
        ]

        connection_list = [
            Connection(1, 2, 3, 0.5, True),
            Connection("connection_number", 2, 3, 0.7, False)
        ]

        genome = Genome(40, node_list, connection_list)
        self.assertEqual(40, genome.seed)
        self.assertEqual(node_list, genome.nodes)
        self.assertEqual(connection_list, genome.connections)
