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

            # Check connection weights
            for connection1, connection2, i in zip(agent1.genome.connections, agent2.genome.connections,
                                                   range(len(agent1.genome.connections))):
                self.assertEqual(connection1.weight, connection2.weight)
                self.assertEqual(connection1.innovation_number, connection2.innovation_number)
                self.assertEqual(i, connection1.innovation_number)

    def test_create_initial_generation_genome(self):
        genome = Genome(1, 20,
                        [Node("abc", NodeType.INPUT, step_function, 0),
                         Node("def", NodeType.OUTPUT, step_function, 1)],
                        [Connection("x", "abc", "def", 1.0, True),
                         Connection("y", "def", "abc", -5, True)])

        generation = gs.create_initial_generation_genome(genome, InnovationNumberGeneratorSingleCore(),
                                                         NeatConfig(population_size=3), seed=1)

        self.assertEqual(3, len(generation.agents))
        self.assertEqual(0, generation.number)
        self.assertEqual(1, len(generation.species_list))
        self.assertEqual(3, len(generation.species_list[0].members))

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

    def test_build_generation_from_genome(self):
        genome = Genome(1, 20,
                        [Node(1, NodeType.INPUT, step_function, 0),
                         Node(2, NodeType.OUTPUT, step_function, 1)],
                        [Connection(1, 1, 2, 1.0, True),
                         Connection(2, 1, 2, -5, True)])

        rnd = np.random.RandomState(1)
        generation = gs._build_generation_from_genome(genome, rnd, NeatConfig(population_size=3))

        self.assertEqual(3, len(generation.agents))
        self.assertEqual(0, generation.number)
        self.assertEqual(1, len(generation.species_list))
        self.assertEqual(3, len(generation.species_list[0].members))

        # First three random numbers
        # 1 - 12710949
        # 2 - 4686059
        # 3 - 6762380
        self.assertEqual(12710949, generation.agents[0].genome.seed)
        self.assertEqual(4686059, generation.agents[1].genome.seed)
        self.assertEqual(6762380, generation.agents[2].genome.seed)

    def test_create_initial_genome(self):
        rnd = np.random.RandomState(1)
        # First value 12710949
        # Second value 4686059
        config = NeatConfig()
        config.connection_min_weight = -10
        config.connection_max_weight = 10
        input_nodes = 100
        output_nodes = 50

        generated_genome1 = gs.create_initial_genome(amount_input_nodes=input_nodes, amount_output_nodes=output_nodes,
                                                     activation_function=step_function, rnd=rnd, config=config,
                                                     generator=InnovationNumberGeneratorSingleCore())

        generated_genome2 = gs.create_initial_genome(amount_input_nodes=input_nodes, amount_output_nodes=output_nodes,
                                                     activation_function=step_function, rnd=rnd, config=config,
                                                     generator=InnovationNumberGeneratorSingleCore())

        rnd = np.random.RandomState(1)

        generated_genome1_1 = gs.create_initial_genome(amount_input_nodes=input_nodes,
                                                       amount_output_nodes=output_nodes,
                                                       activation_function=step_function, rnd=rnd, config=config,
                                                       generator=InnovationNumberGeneratorSingleCore())

        self.assertEqual(12710949, generated_genome1.seed)
        self.assertEqual(4686059, generated_genome2.seed)
        self.assertEqual(12710949, generated_genome1_1.seed)

        self.assertEqual(150, len(generated_genome1.nodes))
        self.assertEqual(5000, len(generated_genome1.connections))

        for node, i in zip(generated_genome1.nodes, range(input_nodes + output_nodes)):
            self.assertEqual(i, node.innovation_number)
            self.assertEqual(step_function, node.activation_function)

            if i < input_nodes:
                self.assertEqual(NodeType.INPUT, node.node_type, msg="Value i={}".format(i))
                self.assertEqual(0, node.x_position)
            else:
                self.assertEqual(NodeType.OUTPUT, node.node_type, msg="Value i={}".format(i))
                self.assertEqual(1, node.x_position)

        last_weight = 0
        for connection, i in zip(generated_genome1.connections, range(input_nodes * output_nodes)):
            self.assertEqual(i, connection.innovation_number)
            self.assertTrue(-10 <= connection.weight <= 10)
            self.assertNotEqual(last_weight, connection.weight)
            last_weight = connection.weight
