from unittest import TestCase

from neat_core.activation_function import step_activation
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service import generation_service as gs
from neat_single_core.agent_id_generator_single_core import AgentIDGeneratorSingleCore
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore
from neat_single_core.species_id_generator_single_core import SpeciesIDGeneratorSingleCore
from utils.reporter import fitness_reporter


class FitnessEvaluationUtilsTest(TestCase):

    def setUp(self) -> None:
        self.config = NeatConfig(population_size=10)
        self.generation = gs.create_initial_generation(3, 2, step_activation,
                                                       InnovationNumberGeneratorSingleCore(),
                                                       SpeciesIDGeneratorSingleCore(),
                                                       AgentIDGeneratorSingleCore(),
                                                       self.config, 1)
        for i, agent in zip(range(len(self.generation.agents)), self.generation.agents):
            agent.fitness = i

    def test_add_generation_fitness_reporter(self):
        reporter = fitness_reporter.FitnessReporter()
        self.assertEqual(0, len(reporter.generations))
        self.assertEqual(0, len(reporter.mean_values))
        self.assertEqual(0, len(reporter.min_values))
        self.assertEqual(0, len(reporter.max_values))
        self.assertEqual(0, len(reporter.std_values))

        reporter = fitness_reporter.add_generation_fitness_reporter(reporter, self.generation)
        self.assertEqual(1, len(reporter.generations))
        self.assertEqual(1, len(reporter.mean_values))
        self.assertEqual(1, len(reporter.min_values))
        self.assertEqual(1, len(reporter.max_values))
        self.assertEqual(1, len(reporter.std_values))

        self.assertEqual(0, reporter.generations[0])
        self.assertEqual((self.config.population_size - 1) / 2, reporter.mean_values[0])
        self.assertEqual(0, reporter.min_values[0])
        self.assertEqual(self.config.population_size - 1, reporter.max_values[0])
        self.assertAlmostEqual(2.8722, reporter.std_values[0], delta=0.0001)
