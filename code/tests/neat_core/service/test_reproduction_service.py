from unittest import TestCase

from neat_core.activation_function import step_function
from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import NodeType, Node
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service.reproduction_service import deep_copy_node, deep_copy_connection, deep_copy_genome, \
    set_new_genome_weights


class ReproductionServiceTest(TestCase):

    def test_deep_copy_genome(self):
        original_genome = Genome(10, 123,
                                 [Node(1, NodeType.INPUT, step_function, 0),
                                  Node("asfaf", NodeType.OUTPUT, step_function, 1)],
                                 [Connection(124, 10, 20, 1.2, True),
                                  Connection("124124", 12, 22, 0.8, False)])

        copied_genome = deep_copy_genome(original_genome)

        # Check if genomes dont have the same id
        self.assertIsNotNone(original_genome)
        self.assertIsNotNone(copied_genome)
        self.assertNotEqual(id(original_genome), id(copied_genome))

        # Compare values
        self.assertEqual(original_genome.id, copied_genome.id)
        self.assertEqual(original_genome.seed, copied_genome.seed)

        for original_node, copied_node in zip(original_genome.nodes, copied_genome.nodes):
            self.compare_nodes(original_node, copied_node)

        for original_connection, copied_connection in zip(original_genome.connections, copied_genome.connections):
            self.compare_connections(original_connection, copied_connection)

    def test_deep_copy_node(self):
        original_node = Node(1, NodeType.INPUT, step_function, 0)
        original_node_str = Node("asfaf", NodeType.OUTPUT, step_function, 1)

        copied_node = deep_copy_node(original_node)
        copied_node_str = deep_copy_node(original_node_str)

        self.compare_nodes(original_node, copied_node)
        self.compare_nodes(original_node_str, copied_node_str)

    def test_deep_copy_connection(self):
        original_connection = Connection(124, 10, 20, 1.2, True)
        original_connection_str = Connection("124124", 12, 22, 0.8, False)

        copied_connection = deep_copy_connection(original_connection)
        copied_connection_str = deep_copy_connection(original_connection_str)

        self.compare_connections(original_connection, copied_connection)
        self.compare_connections(original_connection_str, copied_connection_str)

    def compare_nodes(self, original_node: Node, copied_node: Node):
        # Check if nodes dont have the same id
        self.assertIsNotNone(copied_node)
        self.assertIsNotNone(copied_node)
        self.assertNotEqual(id(original_node), id(copied_node))

        # Check content
        self.assertEqual(original_node.innovation_number, copied_node.innovation_number)
        self.assertEqual(original_node.node_type, copied_node.node_type, )
        self.assertEqual(original_node.activation_function, copied_node.activation_function)
        self.assertEqual(original_node.x_position, copied_node.x_position)

    def compare_connections(self, original_connection: Connection, copied_connection: Connection):
        # Check if connections dont have the same id
        self.assertIsNotNone(original_connection)
        self.assertIsNotNone(copied_connection)
        self.assertNotEqual(id(original_connection), id(copied_connection))

        # Check content
        self.assertEqual(original_connection.innovation_number, copied_connection.innovation_number)
        self.assertEqual(original_connection.input_node, copied_connection.input_node)
        self.assertEqual(original_connection.output_node, copied_connection.output_node)
        self.assertEqual(original_connection.weight, copied_connection.weight)
        self.assertEqual(original_connection.enabled, copied_connection.enabled)

    def test_set_new_genome_weights(self):
        original_genome = Genome(10, 123,
                                 [Node(1, NodeType.INPUT, step_function, 0)],
                                 [Connection(124, 10, 20, 1.2, True),
                                  Connection("124124", 12, 22, 0.8, False)])

        config = NeatConfig(connection_min_weight=-10, connection_max_weight=10)
        new_genome = set_new_genome_weights(original_genome, seed=2, config=config)
        # First 3 random values
        # 1. -1.2801019571599248
        # 2. -9.481475363442174
        # 3. -1.293552147634463

        self.assertAlmostEqual(-1.2801019571599248, new_genome.connections[0].weight, delta=0.00000001)
        self.assertAlmostEqual(-9.481475363442174, new_genome.connections[1].weight, delta=0.00000001)
