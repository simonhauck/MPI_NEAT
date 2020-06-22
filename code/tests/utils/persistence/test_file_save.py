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
        self.assertEqual(self.genome.seed, loaded_genome.seed)
        for node, loaded_node in zip(self.genome.nodes, loaded_genome.nodes):
            self.assertEqual(node.innovation_number, loaded_node.innovation_number)
            self.assertEqual(node.node_type, loaded_node.node_type)
            self.assertEqual(node.bias, loaded_node.bias)
            self.assertEqual(node.activation_function, loaded_node.activation_function)
            self.assertEqual(node.x_position, loaded_node.x_position)

        for connection, loaded_connection in zip(self.genome.connections, loaded_genome.connections):
            self.assertEqual(connection.innovation_number, loaded_connection.innovation_number)
            self.assertEqual(connection.input_node, loaded_connection.input_node)
            self.assertEqual(connection.output_node, loaded_connection.output_node)
            self.assertEqual(connection.weight, loaded_connection.weight)
            self.assertEqual(connection.enabled, loaded_connection.enabled)
