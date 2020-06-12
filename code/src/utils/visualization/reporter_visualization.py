import matplotlib.pyplot as plt

from utils.reporter.fitness_reporter import FitnessReporter
from utils.reporter.species_reporter import SpeciesReporter


def plot_fitness_reporter(reporter: FitnessReporter, plot: bool = False) -> None:
    """
    Prepare/Plot the given reporter
    :param reporter: reporter that holds the fitness values
    :param plot true, if the data should be plotted directly, when false, the plot method must be invoked manually
    :return: None
    """
    plt.title("Fitness values")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    plt.plot(reporter.generations, reporter.max_values, label="max")
    plt.plot(reporter.generations, reporter.mean_values, label="mean")
    plt.plot(reporter.generations, reporter.min_values, label="min")
    plt.legend()

    if plot:
        plt.show()


def plot_species_reporter(reporter: SpeciesReporter, plot: bool = False) -> None:
    x = range(reporter.min_generation, reporter.max_generation + 1)
    y = []

    sorted_id_list = sorted(list(reporter.species_size_dict.keys()))

    for species_id in sorted_id_list:
        generation_numbers, member_size = reporter.species_size_dict[species_id]

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
