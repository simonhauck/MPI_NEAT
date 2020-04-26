from unittest import TestCase

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType


class GenomeTest(TestCase):

    def test_genome(self):
        genome_empty = Genome(1, 20)
        self.assertEqual(1, genome_empty.id)
        self.assertEqual(20, genome_empty.seed)
        self.assertEqual([], genome_empty.nodes)

        node_list = [
            Node(2, NodeType.INPUT, lambda x: x + 1, 0),
            Node("node_number", NodeType.OUTPUT, lambda x: x + 1, 1)
        ]

        connection_list = [
            Connection(1, 2, 3, 0.5, True),
            Connection("connection_number", 2, 3, 0.7, False)
        ]

        genome = Genome(2, 40, node_list, connection_list)
        self.assertEqual(2, genome.id)
        self.assertEqual(40, genome.seed)
        self.assertEqual(node_list, genome.nodes)
        self.assertEqual(connection_list, genome.connections)
