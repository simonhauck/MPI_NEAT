from unittest import TestCase

import numpy as np

from neat_core.activation_function import step_function, modified_sigmoid_function
from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import NodeType, Node
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service.reproduction_service import deep_copy_node, deep_copy_connection, deep_copy_genome, \
    set_new_genome_weights, mutate_weights, mutate_add_connection, mutate_add_node, cross_over, mutate_bias, \
    set_new_genome_bias
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class ReproductionServiceTest(TestCase):

    def setUp(self) -> None:
        self.inn_generator = InnovationNumberGeneratorSingleCore()
        self.rnd = np.random.RandomState(1)

        self.node_input1 = Node(self.inn_generator.get_node_innovation_number(), NodeType.INPUT, 0, step_function, 0)
        self.node_input2 = Node(self.inn_generator.get_node_innovation_number(), NodeType.INPUT, 0, step_function, 0)
        self.node_output1 = Node(self.inn_generator.get_node_innovation_number(), NodeType.OUTPUT, 1.2, step_function,
                                 1)
        self.node_hidden1 = Node(self.inn_generator.get_node_innovation_number(), NodeType.HIDDEN, 1.3, step_function,
                                 0.5)
        self.nodes = [self.node_input1, self.node_input2, self.node_output1, self.node_hidden1]

        self.connection1 = Connection(self.inn_generator.get_connection_innovation_number(),
                                      input_node=self.node_input1.innovation_number,
                                      output_node=self.node_output1.innovation_number, weight=-2.1, enabled=True)
        self.connection2 = Connection(self.inn_generator.get_connection_innovation_number(),
                                      input_node=self.node_input2.innovation_number,
                                      output_node=self.node_output1.innovation_number, weight=-1.2, enabled=True)
        self.connection3 = Connection(self.inn_generator.get_connection_innovation_number(),
                                      input_node=self.node_input1.innovation_number,
                                      output_node=self.node_hidden1.innovation_number, weight=0.6, enabled=True)
        self.connection4 = Connection(self.inn_generator.get_connection_innovation_number(),
                                      input_node=self.node_hidden1.innovation_number,
                                      output_node=self.node_output1.innovation_number, weight=0, enabled=False)
        self.connections = [self.connection1, self.connection2, self.connection3, self.connection4]

        self.genome = Genome(1, self.nodes, connections=self.connections)

    def test_deep_copy_genome(self):
        original_genome = Genome(123,
                                 [Node(1, NodeType.INPUT, 1.1, step_function, 0),
                                  Node("asfaf", NodeType.OUTPUT, 1.2, step_function, 1)],
                                 [Connection(124, 10, 20, 1.2, True),
                                  Connection("124124", 12, 22, 0.8, False)])

        copied_genome = deep_copy_genome(original_genome)

        # Check if genomes dont have the same id
        self.assertIsNotNone(original_genome)
        self.assertIsNotNone(copied_genome)
        self.assertNotEqual(id(original_genome), id(copied_genome))

        # Compare values
        self.assertEqual(original_genome.seed, copied_genome.seed)

        for original_node, copied_node in zip(original_genome.nodes, copied_genome.nodes):
            self.compare_nodes(original_node, copied_node)

        for original_connection, copied_connection in zip(original_genome.connections, copied_genome.connections):
            self.compare_connections(original_connection, copied_connection)

    def test_deep_copy_node(self):
        original_node = Node(1, NodeType.INPUT, 1.1, step_function, 0)
        original_node_str = Node("asfaf", NodeType.OUTPUT, 1.2, step_function, 1)

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
        self.assertEqual(original_node.node_type, copied_node.node_type)
        self.assertEqual(original_node.bias, copied_node.bias)
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
        original_genome = Genome(123,
                                 [Node(1, NodeType.INPUT, 1.1, step_function, 0)],
                                 [Connection(124, 10, 20, 1.2, True),
                                  Connection("124124", 12, 22, 0.8, False)])

        config = NeatConfig(connection_min_weight=-10, connection_max_weight=10)
        new_genome = set_new_genome_weights(original_genome, np.random.RandomState(2), config=config)
        # First 3 random values
        # 1. -1.2801019571599248
        # 2. -9.481475363442174
        # 3. -1.293552147634463

        self.assertAlmostEqual(-1.2801019571599248, new_genome.connections[0].weight, delta=0.00000001)
        self.assertAlmostEqual(-9.481475363442174, new_genome.connections[1].weight, delta=0.00000001)

    def test_set_new_genome_bias(self):
        config = NeatConfig(bias_min=-3, bias_max=3)
        modified_genome = set_new_genome_bias(self.genome, self.rnd, config)

        # Input nodes should be skipped
        self.assertEqual(0, modified_genome.nodes[0].bias)
        self.assertEqual(0, modified_genome.nodes[1].bias)
        self.assertAlmostEqual(-0.4978679717845562, modified_genome.nodes[2].bias, delta=0.0000001)
        self.assertAlmostEqual(1.3219469606529488, modified_genome.nodes[3].bias, delta=0.0000001)

    def test_mutate_weights(self):

        config = NeatConfig(probability_weight_mutation=0.6,
                            probability_random_weight_mutation=0.5,
                            connection_min_weight=-3,
                            connection_max_weight=3,
                            weight_mutation_max_change=1)

        # Random values for with seed 1
        # First connection
        # rnd.uniform(0, 1) = 0.417022004702574 -> Mutate yes
        # rnd.uniform(0, 1) = 0.7203244934421581 -> Perturb weight
        # rnd.uniform(-1, 1) = -0.9997712503653102 -> Value to be subtracted (and clamped)
        # Second connection
        # rnd.uniform(0, 1) = 0.30233257263183977 -> Mutate yes
        # rnd.uniform(0, 1) = 0.14675589081711304 -> Random weight
        # rnd.uniform(-3, 3) = -2.445968431387213 -> new random weight
        # Third connection
        # rnd.uniform(0, 1) = 0.1862602113776709 -> Mutate yes
        # rnd.uniform(0, 1) = 0.34556072704304774 -> Random weight
        # rnd.uniform(-3, 3) = -0.6193951546159804 -> new random weight
        # Fourth connection
        # rnd.uniform(0, 1) = 0.538816734003357 -> Mutate yes
        # rnd.uniform(0, 1) = 0.4191945144032948 -> Random weight
        # rnd.uniform(-3, 3) = 1.1113170023805568 -> new random weight

        new_genome = mutate_weights(self.genome, self.rnd, config)

        # Same object
        self.assertEqual(self.genome, new_genome)
        self.assertEqual(config.connection_min_weight, self.connection1.weight)
        self.assertAlmostEqual(-2.445968431387213, self.connection2.weight, delta=0.000000000001)
        self.assertAlmostEqual(-0.6193951546159804, self.connection3.weight, delta=0.000000000001)
        self.assertAlmostEqual(1.1113170023805568, self.connection4.weight, delta=0.000000000001)

    def test_mutate_bias(self):
        config = NeatConfig(probability_bias_mutation=0.6,
                            bias_max=3,
                            bias_min=-3,
                            probability_random_bias_mutation=0.5,
                            bias_mutation_max_change=1)

        # rnd.uniform(0, 1) = 0.417022004702574 -> Mutate
        # rnd.uniform(0, 1) = 0.7203244934421581 -> Perturb
        # rnd.uniform(-1, 1) = -0.9997712503653102 -> Substract form bias
        # rnd.uniform(0, 1) = 0.30233257263183977 -> Mutate
        # rnd.uniform(0, 1) = 0.14675589081711304 -> Random weight
        # rnd.uniform(-1, 1) = -2.445968431387213 -> new random weight

        old_output_bias = self.node_output1.bias

        new_genome = mutate_bias(self.genome, self.rnd, config)
        self.assertAlmostEqual(old_output_bias - 0.9997712503653102, new_genome.nodes[2].bias, delta=0.000000001)
        self.assertAlmostEqual(-2.445968431387213, new_genome.nodes[3].bias, delta=0.0000000001)

    def test_mutate_add_connection(self):
        config = NeatConfig(connection_min_weight=-3,
                            connection_max_weight=3,
                            allow_recurrent_connections=True,
                            probability_mutate_add_connection=0.4,
                            mutate_connection_tries=2)

        self.genome.connections = []

        # Random values for with seed 1
        # rnd.uniform(0, 1) = 0.417022004702574 -> No mutation
        _, conn = mutate_add_connection(self.genome, self.rnd, self.inn_generator, config)
        self.assertEqual(0, len(self.genome.connections))
        self.assertIsNone(conn)

        # Set higher config value, reset random
        config.probability_mutate_add_connection = 1
        self.rnd = np.random.RandomState(1)
        # Random values for with seed 1
        # rnd.uniform(0, 1) = 0.417022004702574 -> Mutate
        # rnd.randint(low=0,high=4) = 0
        # rnd.randint(low=0,high=2) = 0
        # rnd.uniform(low=-3, high=3) = -2.9993137510959307
        _, new_con1 = mutate_add_connection(self.genome, self.rnd, self.inn_generator, config)
        self.assertEqual(1, len(self.genome.connections))
        self.assertEqual(4, new_con1.innovation_number)
        self.assertEqual(self.node_input1.innovation_number, new_con1.input_node)
        self.assertEqual(self.node_hidden1.innovation_number, new_con1.output_node)
        self.assertAlmostEqual(-2.9993137510959307, new_con1.weight, delta=0.0000000001)

        # Random values
        # rnd.uniform(0, 1) = 0.00011437481734488664 -> Mutate
        # rnd.randint(low=0,high=4) = 3
        # rnd.randint(low=0,high=2) = 0
        # rnd.uniform(low=-3, high=3) = -2.445968431387213
        _, new_con2 = mutate_add_connection(self.genome, self.rnd, self.inn_generator, config)
        self.assertEqual(2, len(self.genome.connections))
        self.assertEqual(5, new_con2.innovation_number)
        self.assertEqual(self.node_output1.innovation_number, new_con2.input_node)
        self.assertEqual(self.node_hidden1.innovation_number, new_con2.output_node)
        self.assertAlmostEqual(-2.445968431387213, new_con2.weight, delta=0.0000000001)

        # Brute force - check if all nodes are connected after 1000 tries
        for _ in range(1000):
            _, _ = mutate_add_connection(self.genome, self.rnd, self.inn_generator, config)

        # Maximum connections with recurrent
        self.assertEqual(8, len(self.genome.connections))

        # Set recurrent to false
        config.allow_recurrent = False
        genome_feed_forward = Genome(1, self.nodes, connections=[])
        for _ in range(1000):
            genome_feed_forward, _ = mutate_add_connection(genome_feed_forward, self.rnd, self.inn_generator, config)
        # Maximum connections with recurrent=false
        self.assertEqual(5, len(genome_feed_forward.connections))

    def test_mutate_add_node(self):
        config = NeatConfig(probability_mutate_add_node=0.1)
        genome, node, con1, con2 = mutate_add_node(self.genome, self.rnd, self.inn_generator, config)

        # Should not mutate
        self.assertEqual(self.genome, genome)
        self.assertIsNone(node)
        self.assertIsNone(con1)
        self.assertIsNone(con2)

        # Set probability high, and reset randomState
        config.probability_mutate_add_node = 1
        self.rnd = np.random.RandomState(1)
        self.node_output1.activation_function = modified_sigmoid_function
        # Random values for with seed 1
        # rnd.uniform(0, 1) = 0.417022004702574 -> Mutate
        # rnd.randint(low=0,high=4) = 0 -> connection1
        # rnd.uniform(0, 1) = -0.9325573593386588 -> Take activation function of out node
        # rnd.uniform(-3, 3) = -2.23125331242386 -> Bias for new node

        node_size_before = len(self.genome.nodes)
        connections_size_before = len(self.connections)

        genome, node, con1, con2 = mutate_add_node(self.genome, self.rnd, self.inn_generator, config)
        # Check if 1 node and 2 connections were added
        self.assertEqual(genome, self.genome)
        self.assertEqual(node_size_before + 1, len(genome.nodes))
        self.assertEqual(connections_size_before + 2, len(genome.connections))

        # Check if old connection was disabled
        self.assertFalse(self.connection1.enabled)

        # Check the generated node
        self.assertEqual(NodeType.HIDDEN, node.node_type)
        self.assertEqual(modified_sigmoid_function, node.activation_function)
        self.assertEqual(4, node.innovation_number)
        self.assertAlmostEqual(-2.23125331242386, node.bias, delta=0.0000000001)
        self.assertEqual(0.5, node.x_position)

        # Check 1 connection
        self.assertEqual(4, con1.innovation_number)
        self.assertEqual(self.connection1.input_node, con1.input_node)
        self.assertEqual(node.innovation_number, con1.output_node)
        self.assertEqual(1, con1.weight)
        self.assertTrue(con1.enabled)

        # Check 2 connection
        self.assertEqual(5, con2.innovation_number)
        self.assertEqual(node.innovation_number, con2.input_node)
        self.assertEqual(self.connection1.output_node, con2.output_node)
        self.assertEqual(self.connection1.weight, con2.weight)
        self.assertTrue(con2.enabled)

    def test_cross_over(self):
        node1_1 = Node(1, NodeType.INPUT, 1.1, step_function, 0)
        node1_2 = Node(2, NodeType.INPUT, 1.2, step_function, 0)
        node1_5 = Node(5, NodeType.OUTPUT, 1.5, step_function, 1)
        node1_7 = Node(7, NodeType.HIDDEN, 1.7, step_function, 0.5)
        nodes1 = [node1_1, node1_2, node1_5, node1_7]

        node2_1 = Node(1, NodeType.INPUT, 2.1, modified_sigmoid_function, 0)
        node2_2 = Node(2, NodeType.INPUT, 2.2, modified_sigmoid_function, 0)
        node2_4 = Node(4, NodeType.OUTPUT, 2.4, modified_sigmoid_function, 1)
        node2_7 = Node(7, NodeType.HIDDEN, 2.7, modified_sigmoid_function, 0.5)
        node2_8 = Node(8, NodeType.HIDDEN, 2.8, modified_sigmoid_function, 0.5)
        nodes2 = [node2_1, node2_2, node2_4, node2_7, node2_8]

        connection1_1 = Connection(innovation_number=1, input_node=1, output_node=2, weight=1.2, enabled=False)
        connection1_2 = Connection(innovation_number=2, input_node=1, output_node=2, weight=1.2, enabled=True)
        connection1_4 = Connection(innovation_number=4, input_node=1, output_node=2, weight=1.2, enabled=True)
        connection1_7 = Connection(innovation_number=7, input_node=1, output_node=2, weight=1.2, enabled=False)
        connections1 = [connection1_1, connection1_2, connection1_4, connection1_7]

        connection2_1 = Connection(innovation_number=1, input_node=1, output_node=2, weight=1.2, enabled=False)
        connection2_2 = Connection(innovation_number=2, input_node=1, output_node=2, weight=1.2, enabled=True)
        connection2_3 = Connection(innovation_number=3, input_node=1, output_node=2, weight=1.2, enabled=True)
        connection2_5 = Connection(innovation_number=5, input_node=1, output_node=2, weight=1.2, enabled=True)
        connection2_7 = Connection(innovation_number=7, input_node=1, output_node=2, weight=1.2, enabled=False)
        connections2 = [connection2_1, connection2_2, connection2_3, connection2_5, connection2_7]

        more_fit_parent = Genome(1, nodes1, connections1)
        less_fit_parent = Genome(2, nodes2, connections2)

        config = NeatConfig(probability_enable_gene=0.31)
        # Random values:
        # 1 rnd.uniform(0, 1) = 0.417022004702574 -> First node (node1_1)
        # 1 rnd.uniform(0, 1) = 0.7203244934421581 -> Second node(node2_2)
        # 1 rnd.uniform(0, 1) = 0.00011437481734488664 -> First node (node7_1)

        # 1 rnd.uniform(0, 1) = 0.30233257263183977 -> Select first connection (connection1_1)
        # 1 rnd.uniform(0, 1) = 0.14675589081711304 -> Re-enable connection (connection1_1)
        # 1 rnd.uniform(0, 1) = 0.0923385947687978 -> Select first connection (connection1_2)
        # 1 rnd.uniform(0, 1) = 0.1862602113776709 -> Select first connection (connection1_7)
        # 1 rnd.uniform(0, 1) = 0.34556072704304774 -> Re-enable connection False (connection1_7)

        child_nodes, child_connections = cross_over(more_fit_parent, less_fit_parent, self.rnd, config)
        # Check nodes
        self.assertEqual(4, len(child_nodes))
        self.compare_nodes(node1_1, child_nodes[0])
        self.compare_nodes(node2_2, child_nodes[1])
        self.compare_nodes(node1_5, child_nodes[2])
        self.compare_nodes(node1_7, child_nodes[3])

        # Check connections
        self.assertEqual(4, len(child_connections))
        connection1_1.enabled = True
        self.compare_connections(connection1_1, child_connections[0])
        self.compare_connections(connection1_2, child_connections[1])
        self.compare_connections(connection1_4, child_connections[2])
        self.compare_connections(connection1_7, child_connections[3])
