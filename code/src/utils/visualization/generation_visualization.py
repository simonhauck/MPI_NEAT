import matplotlib.pyplot as plt
import numpy as np

from neat_core.models.generation import Generation


class PlotData(object):

    def __init__(self) -> None:
        self.generations: np.ndarray = np.array([])
        self.min_values: np.ndarray = np.array([])
        self.max_values: np.ndarray = np.array([])
        self.mean_values: np.ndarray = np.array([])
        self.std_values: np.ndarray = np.array([])

    def add_generation(self, generation: Generation) -> None:
        """
        Add a generation to the data, that can be plotted
        :param generation:
        :return: None
        """
        self.generations = np.append(self.generations, generation.number)

        fitness_values = np.array([agent.fitness for agent in generation.agents])
        min = np.min(fitness_values)
        max = np.max(fitness_values)
        mean = np.mean(fitness_values)
        std = np.std(fitness_values)

        self.min_values = np.append(self.min_values, min)
        self.max_values = np.append(self.max_values, max)
        self.mean_values = np.append(self.mean_values, mean)
        self.std_values = np.append(self.std_values, std)


def plot_fitness_values(plot_data: PlotData) -> None:
    """
    Plot the fitness values with the given plot_data object
    :param plot_data: objects that holds the aggregated data
    :return: None
    """
    plt.title("Fitness values")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    plt.plot(plot_data.generations, plot_data.max_values, label="max")
    plt.plot(plot_data.generations, plot_data.mean_values, label="mean")
    plt.plot(plot_data.generations, plot_data.min_values, label="min")
    plt.legend()
