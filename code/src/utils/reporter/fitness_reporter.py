from typing import List

import numpy as np

from neat_core.models.generation import Generation
from neat_core.optimizer.neat_reporter import NeatReporter


class FitnessReporterData(object):

    def __init__(self) -> None:
        self.generations: np.ndarray = np.array([])
        self.min_values: np.ndarray = np.array([])
        self.max_values: np.ndarray = np.array([])
        self.mean_values: np.ndarray = np.array([])
        self.std_values: np.ndarray = np.array([])


class FitnessReporter(NeatReporter):

    def __init__(self) -> None:
        self.data = FitnessReporterData()

    def on_generation_evaluation_end(self, generation: Generation, reporters: List[NeatReporter]) -> None:
        """
        Add a generation to the fitness reporter. The reporter extracts the min, max and mean fitness values and calculates
        the standard deviation
        :param generation: the generation that will be added
        :param reporters: a list with all reporters
        :return: None
        """
        self.data.generations = np.append(self.data.generations, generation.number)

        fitness_values = np.array([agent.fitness for agent in generation.agents])
        min = np.min(fitness_values)
        max = np.max(fitness_values)
        mean = np.mean(fitness_values)
        std = np.std(fitness_values)

        self.data.min_values = np.append(self.data.min_values, min)
        self.data.max_values = np.append(self.data.max_values, max)
        self.data.mean_values = np.append(self.data.mean_values, mean)
        self.data.std_values = np.append(self.data.std_values, std)

    def store_data(self) -> (bool, str, object):
        return True, "fitness_reporter", self.data
