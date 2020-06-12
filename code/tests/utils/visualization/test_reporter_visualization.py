from unittest import TestCase

from neat_core.activation_function import step_activation
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service import generation_service as gs
from neat_single_core.agent_id_generator_single_core import AgentIDGeneratorSingleCore
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore
from neat_single_core.species_id_generator_single_core import SpeciesIDGeneratorSingleCore
from utils.reporter import fitness_reporter
from utils.visualization import reporter_visualization


class ReporterVisualizationTest(TestCase):

    def setUp(self) -> None:
        self.config = NeatConfig(population_size=10)
        self.generation = gs.create_initial_generation(3, 2, step_activation,
                                                       InnovationNumberGeneratorSingleCore(),
                                                       SpeciesIDGeneratorSingleCore(),
                                                       AgentIDGeneratorSingleCore(),
                                                       self.config, 1)
        for i, agent in zip(range(len(self.generation.agents)), self.generation.agents):
            agent.fitness = i

    def test_plot_fitness_reporter(self):
        reporter = fitness_reporter.FitnessReporter()
        reporter = fitness_reporter.add_generation_fitness_reporter(reporter, self.generation)

        # Test if error occurs
        reporter_visualization.plot_fitness_reporter(reporter, plot=True)
