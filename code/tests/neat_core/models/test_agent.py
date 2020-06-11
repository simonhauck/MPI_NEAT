from unittest import TestCase

from neat_core.activation_function import modified_sigmoid_activation
from neat_core.models.agent import Agent
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service.generation_service import create_genome_structure
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class AgentTest(TestCase):

    def test_agent(self):
        genome = create_genome_structure(5, 2, modified_sigmoid_activation, NeatConfig(),
                                         InnovationNumberGeneratorSingleCore())
        agent = Agent(1, genome)

        self.assertEqual(1, agent.id)
        self.assertEqual(genome, agent.genome)
        self.assertIsNone(agent.neural_network)
        self.assertEqual(0, agent.fitness)
        self.assertEqual(0, agent.adjusted_fitness)
