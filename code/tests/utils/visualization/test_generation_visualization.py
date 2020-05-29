from unittest import TestCase

import numpy as np

from neat_core.activation_function import step_function
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service import generation_service as gs
from neat_single_core.agent_id_generator_single_core import AgentIDGeneratorSingleCore
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore
from neat_single_core.species_id_generator_single_core import SpeciesIDGeneratorSingleCore
from utils.visualization.generation_visualization import PlotData


class GenerationVisualizationTest(TestCase):

    def setUp(self) -> None:
        self.config = NeatConfig(population_size=150)
        self.generation1 = gs.create_initial_generation(3, 2, step_function,
                                                        InnovationNumberGeneratorSingleCore(),
                                                        SpeciesIDGeneratorSingleCore(),
                                                        AgentIDGeneratorSingleCore(),
                                                        self.config, 1)
        for i, agent in zip(range(len(self.generation1.agents)), self.generation1.agents):
            agent.fitness = i

    def test_add_generation(self):
        max_fitness = self.config.population_size - 1
        min_fitness = 0
        mean = np.mean(range(self.config.population_size))
        std = np.std(range(self.config.population_size))

        plot_data = PlotData()

        # Check data before adding
        self.assertEqual(0, len(plot_data.generations))
        self.assertEqual(0, len(plot_data.min_values))
        self.assertEqual(0, len(plot_data.max_values))
        self.assertEqual(0, len(plot_data.mean_values))
        self.assertEqual(0, len(plot_data.std_values))

        plot_data.add_generation(self.generation1)
        # Check len
        self.assertEqual(1, len(plot_data.generations))
        self.assertEqual(1, len(plot_data.min_values))
        self.assertEqual(1, len(plot_data.max_values))
        self.assertEqual(1, len(plot_data.mean_values))
        self.assertEqual(1, len(plot_data.std_values))

        # Check value
        self.assertEqual(0, plot_data.generations[0])
        self.assertEqual(min_fitness, plot_data.min_values[0])
        self.assertEqual(max_fitness, plot_data.max_values[0])
        self.assertEqual(mean, plot_data.mean_values[0])
        self.assertEqual(std, plot_data.std_values[0])
