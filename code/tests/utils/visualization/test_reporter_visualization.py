from unittest import TestCase

import numpy as np

from neat_core.optimizer.neat_config import NeatConfig
from utils.reporter import fitness_reporter, species_reporter, time_reporter
from utils.visualization import reporter_visualization


class ReporterVisualizationTest(TestCase):

    def setUp(self) -> None:
        self.config = NeatConfig(population_size=10)

    def test_plot_fitness_reporter(self):
        data = fitness_reporter.FitnessReporterData()
        data.generations = np.array([1, 2, 3, 4])
        data.max_values = np.array([5, 6, 7, 8])
        data.min_values = np.array([1, 1, 1, 1])
        data.mean_values = np.array([3, 4, 4, 5])
        data.std_values = np.array([0.5, 0.5, 0.5, 0.5])

        # Test if error occurs
        reporter_visualization.plot_fitness_reporter(data, plot=True)

    def test_plot_species_reporter(self):
        data = species_reporter.SpeciesReporterData()
        data.min_generation = 2
        data.max_generation = 4
        data.species_size_dict = {
            1: ([2, 3, 4], [10, 6, 7]),
            2: ([3], [2]),
            3: ([3, 4], [2, 3])
        }

        # Test if error occurs
        reporter_visualization.plot_species_reporter(data, plot=True)

    def test_plot_time_reporter(self):
        data1 = time_reporter.TimeReporterEntry(1, 1, 2, 4)
        data2 = time_reporter.TimeReporterEntry(2, 2, 5, 8)
        data3 = time_reporter.TimeReporterEntry(3, 3, 6, 12)
        data4 = time_reporter.TimeReporterEntry(4, 4, 7, 15)

        data = [data1, data3, data4, data2]

        # Test if error occurs
        reporter_visualization.plot_time_reporter(data, plot=True)
