import matplotlib.pyplot as plt

from utils.reporter.fitness_reporter import FitnessReporter


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
