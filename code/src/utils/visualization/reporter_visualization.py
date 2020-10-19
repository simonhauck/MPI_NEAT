from typing import List

import matplotlib.pyplot as plt

from utils.reporter.fitness_reporter import FitnessReporterData
from utils.reporter.species_reporter import SpeciesReporterData
from utils.reporter.time_reporter import TimeReporterEntry

font = {'family': 'normal',
        'size': 13}

plt.rc('font', **font)


def plot_fitness_reporter(data: FitnessReporterData, plot: bool = False) -> None:
    """
    Prepare/Plot the given reporter
    :param data: reporter that holds the fitness values
    :param plot true, if the data should be plotted directly, when false, the plot method must be invoked manually
    :return: None
    """
    plt.title("Fitness values")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    plt.plot(data.generations, data.max_values, label="max")
    plt.plot(data.generations, data.mean_values, label="mean")
    plt.plot(data.generations, data.min_values, label="min")
    plt.legend()

    plt.gca().locator_params(nbins=11)

    if plot:
        plt.show()


def plot_species_reporter(data: SpeciesReporterData, plot: bool = False) -> None:
    """
    Plot the species reporter in a stacked area plot
    :param data: the reporter with the species data
    :param plot: true, if the plt.show() method should be invoked
    :return: None
    """
    x = range(data.min_generation, data.max_generation + 1)
    y = []

    sorted_id_list = sorted(list(data.species_size_dict.keys()))

    for species_id in sorted_id_list:
        generation_numbers, member_size = data.species_size_dict[species_id]

        species_size_values = []

        for generation_number in x:
            try:
                size_index = generation_numbers.index(generation_number)
                size = member_size[size_index]
            except ValueError:
                size = 0

            species_size_values.append(size)

        y.append(species_size_values)

    plt.stackplot(x, y)

    if plot:
        plt.show()


def plot_time_reporter(data: List[TimeReporterEntry], plot: bool = False) -> None:
    # Sort data if something is wrong
    data = sorted(data, key=lambda entry: entry.generation)

    generations = [entry.generation for entry in data]
    evaluation_times = [entry.evaluation_time for entry in data]
    reproduction_times = [entry.reproduction_time for entry in data]
    compose_offspring_times = [entry.compose_offspring_time for entry in data]

    bottom_compose_offspring_times = [evaluation_time + reproduction_time for evaluation_time, reproduction_time in
                                      zip(evaluation_times, reproduction_times)]

    bar1 = plt.bar(generations, evaluation_times)
    bar2 = plt.bar(generations, reproduction_times, bottom=evaluation_times)
    bar3 = plt.bar(generations, compose_offspring_times, bottom=bottom_compose_offspring_times)

    plt.title("Execution Time")
    plt.ylabel("Time (s)")
    plt.xlabel("Generations")
    plt.legend((bar1[0], bar2[0], bar3[0]), ('Evaluation Time', 'Reproduction Time', 'Compose Offspring Time'))

    plt.gca().locator_params(nbins=11)

    if plot:
        plt.show()
