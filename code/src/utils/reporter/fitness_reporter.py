import numpy as np

from neat_core.models.generation import Generation


class FitnessReporter(object):

    def __init__(self) -> None:
        self.generations: np.ndarray = np.array([])
        self.min_values: np.ndarray = np.array([])
        self.max_values: np.ndarray = np.array([])
        self.mean_values: np.ndarray = np.array([])
        self.std_values: np.ndarray = np.array([])


def add_generation_fitness_reporter(reporter: FitnessReporter, generation: Generation) -> FitnessReporter:
    """
    Add a generation to the fitness reporter. The reporter extracts the min, max and mean fitness values and calculates
    the standard deviation
    :param reporter: the reporter, to which the data will be added
    :param generation: the generation data that should be added
    :return: the updated fitness reporter
    """
    reporter.generations = np.append(reporter.generations, generation.number)

    fitness_values = np.array([agent.fitness for agent in generation.agents])
    min = np.min(fitness_values)
    max = np.max(fitness_values)
    mean = np.mean(fitness_values)
    std = np.std(fitness_values)

    reporter.min_values = np.append(reporter.min_values, min)
    reporter.max_values = np.append(reporter.max_values, max)
    reporter.mean_values = np.append(reporter.mean_values, mean)
    reporter.std_values = np.append(reporter.std_values, std)

    return reporter
