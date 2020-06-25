import time
from unittest import TestCase

import numpy as np

from neat_core.activation_function import step_activation, modified_sigmoid_activation
from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from utils.persistance import file_save


class FileSaveTest(TestCase):

    def setUp(self) -> None:
        seed = np.random.RandomState().randint(2 ** 24)
        rnd = np.random.RandomState(seed)

        self.genome = Genome(seed,
                             [
                                 Node(1, NodeType.INPUT, rnd.uniform(-5, 5), step_activation, 0.5),
                                 Node(2, NodeType.OUTPUT, rnd.uniform(-5, 5), modified_sigmoid_activation, 0.5),
                                 Node(3, NodeType.HIDDEN, rnd.uniform(-5, 5), step_activation, 0.5),
                                 Node(4, NodeType.HIDDEN, rnd.uniform(-5, 5), step_activation, 0.5),
                             ],
                             [
                                 Connection(1, 1, 2, rnd.uniform(-5, 5), True),
                                 Connection(2, 1, 4, rnd.uniform(-5, 5), True),
                                 Connection(3, 1, 3, rnd.uniform(-5, 5), True),
                                 Connection(4, 3, 2, rnd.uniform(-5, 5), True),
                                 Connection(5, 4, 2, rnd.uniform(-5, 5), True),
                                 Connection(6, 2, 2, rnd.uniform(-5, 5), False),
                             ])

    def test_save_genome_file(self):
        current_time = time.time_ns()
        filename = "/tmp/test_genome_save_" + str(current_time) + ".bak"
        file_save.save_genome_file(filename, self.genome)

    def test_load_genome_file(self):
        current_time = time.time_ns()
        filename = "/tmp/test_genome_save_" + str(current_time) + ".bak"
        file_save.save_genome_file(filename, self.genome)

        loaded_genome = file_save.load_genome_file(filename)

        self.compare_genomes(self.genome, loaded_genome)

    def compare_genomes(self, g1: Genome, g2: Genome):
        self.assertEqual(g1.seed, g2.seed)
        for g1_node, g2_node in zip(g1.nodes, g2.nodes):
            self.assertEqual(g1_node.innovation_number, g2_node.innovation_number)
            self.assertEqual(g1_node.node_type, g2_node.node_type)
            self.assertEqual(g1_node.bias, g2_node.bias)
            self.assertEqual(g1_node.activation_function, g2_node.activation_function)
            self.assertEqual(g1_node.x_position, g2_node.x_position)

        for g1_connection, g2_connection in zip(g1.connections, g2.connections):
            self.assertEqual(g1_connection.innovation_number, g2_connection.innovation_number)
            self.assertEqual(g1_connection.input_node, g2_connection.input_node)
            self.assertEqual(g1_connection.output_node, g2_connection.output_node)
            self.assertEqual(g1_connection.weight, g2_connection.weight)
            self.assertEqual(g1_connection.enabled, g2_connection.enabled)
