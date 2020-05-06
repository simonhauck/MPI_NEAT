from unittest import TestCase

import numpy as np

from neat_core.activation_function import step_function
from neat_core.models.node import NodeType
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service.generation_service import create_initial_genome
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class ReproductionServiceTest(TestCase):

    def test_create_initial_genome(self):
        rnd = np.random.RandomState(1)
        # First value 12710949
        # Second value 4686059
        config = NeatConfig()
        config.connection_min_weight = -10
        config.connection_max_weight = 10
        input_nodes = 100
        output_nodes = 50

        generated_genome1 = create_initial_genome(amount_input_nodes=input_nodes, amount_output_nodes=output_nodes,
                                                  activation_function=step_function, rnd=rnd, config=config,
                                                  generator=InnovationNumberGeneratorSingleCore())

        generated_genome2 = create_initial_genome(amount_input_nodes=input_nodes, amount_output_nodes=output_nodes,
                                                  activation_function=step_function, rnd=rnd, config=config,
                                                  generator=InnovationNumberGeneratorSingleCore())

        rnd = np.random.RandomState(1)

        generated_genome1_1 = create_initial_genome(amount_input_nodes=input_nodes, amount_output_nodes=output_nodes,
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
