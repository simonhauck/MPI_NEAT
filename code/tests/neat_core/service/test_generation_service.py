from unittest import TestCase

import numpy as np

import neat_core.service.generation_service as gs
from neat_core.activation_function import step_function
from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.neat_config import NeatConfig
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class GenerationServiceTest(TestCase):

    def test_create_initial_generation(self):
        generation1 = gs.create_initial_generation(10, 3, step_function, InnovationNumberGeneratorSingleCore(),
                                                   NeatConfig(population_size=150, connection_max_weight=10,
                                                              connection_min_weight=-10), seed=1)

        generation2 = gs.create_initial_generation(10, 3, step_function, InnovationNumberGeneratorSingleCore(),
                                                   NeatConfig(population_size=150, connection_max_weight=10,
                                                              connection_min_weight=-10), seed=1)

        self.assertEqual(0, generation1.number)
        self.assertEqual(0, generation2.number)
        self.assertEqual(150, len(generation1.agents))

        # Compare if generations and generated genomes are the same
        for agent1, agent2 in zip(generation1.agents, generation2.agents):
            self.assertEqual(agent1.genome.seed, agent2.genome.seed)

            for node1, node2, i in zip(agent1.genome.nodes, agent2.genome.nodes, range(len(agent1.genome.nodes))):
                self.assertEqual(node1.innovation_number, node2.innovation_number)
                self.assertEqual(i, node1.innovation_number)
                self.assertEqual(node1.bias, node2.bias)

                if node1.node_type != NodeType.INPUT:
                    self.assertNotAlmostEqual(0, node1.bias, delta=0.000000000001)

            # Check connection weights
            for connection1, connection2, i in zip(agent1.genome.connections, agent2.genome.connections,
                                                   range(len(agent1.genome.connections))):
                self.assertEqual(connection1.weight, connection2.weight)
                self.assertEqual(connection1.innovation_number, connection2.innovation_number)
                self.assertEqual(i, connection1.innovation_number)
                self.assertNotAlmostEqual(0, connection1.weight, delta=0.000000000001)

    def test_create_initial_generation_genome(self):
        genome = Genome(1, 20,
                        [Node("abc", NodeType.INPUT, 0.3, step_function, 0),
                         Node("def", NodeType.OUTPUT, 0.4, step_function, 1)],
                        [Connection("x", "abc", "def", 1.0, True),
                         Connection("y", "def", "abc", -5, True)])

        generation = gs.create_initial_generation_genome(genome, InnovationNumberGeneratorSingleCore(),
                                                         NeatConfig(population_size=3), seed=1)

        self.assertEqual(3, len(generation.agents))
        self.assertEqual(0, generation.number)
        self.assertEqual(1, len(generation.species_list))
        self.assertEqual(3, len(generation.species_list[0].members))
        self.assertEqual(1, generation.seed)

        # First three random numbers
        # 1 - 12710949
        # 2 - 4686059
        # 3 - 6762380
        self.assertEqual(12710949, generation.agents[0].genome.seed)
        self.assertEqual(4686059, generation.agents[1].genome.seed)
        self.assertEqual(6762380, generation.agents[2].genome.seed)

        for node1, node2, node3, i in zip(generation.agents[0].genome.nodes,
                                          generation.agents[1].genome.nodes,
                                          generation.agents[2].genome.nodes,
                                          range(3)):
            self.assertEqual(node1.innovation_number, node2.innovation_number)
            self.assertEqual(node2.innovation_number, node3.innovation_number)
            self.assertEqual(node3.innovation_number, i)

        for connection1, connection2, connection3, i in zip(generation.agents[0].genome.connections,
                                                            generation.agents[1].genome.connections,
                                                            generation.agents[2].genome.connections,
                                                            range(3)):
            self.assertEqual(connection1.innovation_number, connection2.innovation_number)
            self.assertEqual(connection2.innovation_number, connection3.innovation_number)
            self.assertEqual(connection3.innovation_number, i)

    def test_randomize_weight_bias(self):
        genome = Genome(1, 20,
                        [Node(1, NodeType.INPUT, 0.0, step_function, 0),
                         Node(2, NodeType.OUTPUT, 0.0, step_function, 1)],
                        [Connection(3, 1, 2, 0, True),
                         Connection(4, 2, 1, 0, True)])

        config = NeatConfig(bias_min=1, bias_max=2, connection_min_weight=3, connection_max_weight=4)
        randomized_genome = gs._randomize_weight_bias(genome, np.random.RandomState(1), config)

        for node in randomized_genome.nodes:
            if node.node_type == NodeType.INPUT:
                self.assertEqual(0, node.bias)
            else:
                self.assertTrue(1 <= node.bias <= 2)

        for conn in randomized_genome.connections:
            self.assertTrue(3 <= conn.weight <= 4)

    def test_build_generation_from_genome(self):
        initial_genome = Genome(1, 20,
                                [Node(1, NodeType.INPUT, 0, step_function, 0),
                                 Node(2, NodeType.OUTPUT, 0.4, step_function, 1)],
                                [Connection(1, 1, 2, 1.0, True),
                                 Connection(2, 1, 2, -5, True)])

        rnd = np.random.RandomState(1)
        generation = gs._build_generation_from_genome(initial_genome, 19, rnd, NeatConfig(population_size=3))

        self.assertEqual(3, len(generation.agents))
        self.assertEqual(0, generation.number)
        self.assertEqual(1, len(generation.species_list))
        self.assertEqual(3, len(generation.species_list[0].members))
        self.assertEqual(19, generation.seed)

        # First three random numbers
        # 1 - 12710949
        # 2 - 4686059
        # 3 - 6762380
        self.assertEqual(12710949, generation.agents[0].genome.seed)
        self.assertEqual(4686059, generation.agents[1].genome.seed)
        self.assertEqual(6762380, generation.agents[2].genome.seed)

        # Check if every weight and bias is not equal to 0
        for agent in generation.agents:
            genome = agent.genome
            for node in genome.nodes:
                # Input nodes can have bias 0
                if node.node_type == NodeType.INPUT:
                    continue
                self.assertNotAlmostEqual(0, node.bias, delta=0.00000000001)

            for conn in genome.connections:
                self.assertNotAlmostEqual(0, conn.weight, delta=0.00000000001)

    def test_create_genome_structure(self):

        config = NeatConfig()
        input_nodes = 100
        output_nodes = 50

        generated_structure = gs.create_genome_structure(amount_input_nodes=input_nodes,
                                                         amount_output_nodes=output_nodes,
                                                         activation_function=step_function, config=config,
                                                         generator=InnovationNumberGeneratorSingleCore())

        self.assertEqual(input_nodes + output_nodes, len(generated_structure.nodes))
        self.assertEqual(input_nodes * output_nodes, len(generated_structure.connections))

        for node, i in zip(generated_structure.nodes, range(input_nodes + output_nodes)):
            self.assertEqual(i, node.innovation_number)
            self.assertEqual(step_function, node.activation_function)
            self.assertEqual(0, node.bias)

            if i < input_nodes:
                self.assertEqual(NodeType.INPUT, node.node_type, msg="Value i={}".format(i))
                self.assertEqual(0, node.x_position)
            else:
                self.assertEqual(NodeType.OUTPUT, node.node_type, msg="Value i={}".format(i))
                self.assertEqual(1, node.x_position)

        for connection, i in zip(generated_structure.connections, range(input_nodes * output_nodes)):
            self.assertEqual(i, connection.innovation_number)
            self.assertEqual(0, connection.weight)
