# import matplotlib.pyplot as plt
import numpy as np

from neat_core.models.generation import Generation


class PlotData(object):

    def __init__(self) -> None:
        self.generations: np.ndarray = np.array([])
        self.min_values: np.ndarray = np.array([])
        self.max_values: np.ndarray = np.array([])
        self.mean_values: np.ndarray = np.array([])
        self.std_values: np.ndarray = np.array([])


def add_generation(plot_data: PlotData, generation: Generation) -> PlotData:
    plot_data.generations = np.append(plot_data.generations, generation.number)

    fitness_values = np.array([agent.fitness for agent in generation.agents])
    min = np.min(fitness_values)
    max = np.max(fitness_values)
    mean = np.mean(fitness_values)
    std = np.std(fitness_values)

    plot_data.min_values = np.append(plot_data.min_values, min)
    plot_data.max_values = np.append(plot_data.max_values, max)
    plot_data.mean_values = np.append(plot_data.mean_values, mean)
    plot_data.std_values = np.append(plot_data.std_values, std)

    return plot_data


def plot_fitness_values(plot_data: PlotData) -> None:
    # plt.errorbar(np.arange(8), plot_data.mean_values, plot_data.std_values, fmt='ok', lw=3)
    # plt.errorbar(np.arange(8), plot_data.mean_values,
    #              [plot_data.mean_values - plot_data.min_values, plot_data.max_values - plot_data.mean_values],
    #              fmt='.k', ecolor='gray', lw=1)
    return
