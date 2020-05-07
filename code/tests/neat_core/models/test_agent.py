from unittest import TestCase

import numpy as np

from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.agent import Agent
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service.generation_service import create_initial_genome
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class AgentTest(TestCase):

    def test_agent(self):
        genome = create_initial_genome(5, 2, modified_sigmoid_function, np.random.RandomState(), NeatConfig(),
                                       InnovationNumberGeneratorSingleCore())
        agent = Agent(genome)

        self.assertEqual(genome, agent.genome)
        self.assertIsNone(agent.neural_network)
        self.assertEqual(0, agent.fitness)
        self.assertEqual(0, agent.adjusted_fitness)
