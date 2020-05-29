from unittest import TestCase

from neat_core.activation_function import step_function
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service import generation_service as gs
from neat_single_core.agent_id_generator_single_core import AgentIDGeneratorSingleCore
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore
from neat_single_core.species_id_generator_single_core import SpeciesIDGeneratorSingleCore
from utils.fitness_evaluation.fitness_evaluation_utils import get_best_agent


class FitnessEvaluationUtilsTest(TestCase):

    def setUp(self) -> None:
        self.config = NeatConfig(population_size=150)
        self.generation = gs.create_initial_generation(3, 2, step_function,
                                                       InnovationNumberGeneratorSingleCore(),
                                                       SpeciesIDGeneratorSingleCore(),
                                                       AgentIDGeneratorSingleCore(),
                                                       self.config, 1)
        for i, agent in zip(range(len(self.generation.agents)), self.generation.agents):
            agent.fitness = i

    def test_get_best_agent(self):
        expected_best_agent = self.generation.agents[len(self.generation.agents) - 1]
        best_agent = get_best_agent(self.generation.agents)

        self.assertEqual(expected_best_agent, best_agent)
        self.assertEqual(self.config.population_size - 1, best_agent.fitness)
