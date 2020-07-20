import argparse
import sys

from loguru import logger

from examples.breakout.breakout import BreakoutOptimizer
from examples.lunar_lander.lunar_lander import LunarLanderOptimizer
from examples.mountain_car.mountain_car import MountainCarOptimizer
from examples.pendulum.pendulum import PendulumOptimizer
from examples.pole_balancing.pole_balancing import PoleBalancingOptimizer
from examples.xor.xor_evaluation import XOROptimizer
from neat_core.models.genome import Genome
from utils.performance_evalation import performance_comparison2 as performance
from utils.persistance import file_save
from utils.visualization import reporter_visualization, genome_visualization, text_visualization

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | {level}   | <level>{message}</level>",
           level="INFO")


def xor_visualization(genome: Genome):
    xor_optimizer = XOROptimizer()
    xor_optimizer.visualize_genome(genome)
    return xor_optimizer.solved_generation_number


def mountain_car_single_visualization(genome: Genome):
    mountain_optimizer = MountainCarOptimizer()
    mountain_optimizer.visualize_genome(genome)
    return 0


def pole_balancing_single_visualization(genome: Genome):
    pole_balancing_optimizer = PoleBalancingOptimizer()
    pole_balancing_optimizer.visualize_genome(genome)
    return 0


def pendulum_single_visualization(genome: Genome):
    pendulum_optimizer = PendulumOptimizer()
    pendulum_optimizer.visualize_genome(genome)
    return 0


def lunar_lander_single_visualization(genome: Genome):
    lunar_lander_optimizer = LunarLanderOptimizer()
    lunar_lander_optimizer.visualize_genome(genome)


def breakout_single_visualization(genome: Genome):
    breakout_optimizer = BreakoutOptimizer()
    breakout_optimizer.visualize_genome(genome)


challenge_dict = {
    "xor": xor_visualization,
    "mountain_car": mountain_car_single_visualization,
    "pole_balancing": pole_balancing_single_visualization,
    "pendulum": pendulum_single_visualization,
    "lunar_lander": lunar_lander_single_visualization,
    "breakout": breakout_single_visualization
}

parser = argparse.ArgumentParser(description="Visualize stored NEAT genomes in the challenges")
parser.add_argument('challenge', type=str, help="Select one example challenge, that should be run",
                    choices=challenge_dict.keys())
parser.add_argument("genome_file", type=str, help="Path to the genome that should be loaded")
parser.add_argument("-r", metavar="--repeat", type=int, default=1, help="Run the same challenge multiple times")
parser.add_argument("-c", metavar="--compare", type=str, default=None, help="Compare the time values to some other run")
parser.add_argument("-n", metavar="number", type=int, default=1, help="Number of used cores in the multi core")

args = parser.parse_args()

challenge = args.challenge
genome_file = args.genome_file
amount_runs = args.r
compare_file = args.c
number_of_cores = args.n

logger.info("Selected challenge: {}, Amount Runs: {}".format(challenge, amount_runs))

if genome_file.endswith(".genome"):
    loaded_genome = file_save.load_genome_file(genome_file)
    reporter_dict = {}
elif genome_file.endswith(".data"):
    loaded_genome, reporter_dict = file_save.load_genome_and_reporter(genome_file)
else:
    logger.warning("No genome found, can't continue")
    exit(-1)

# Visualize graphs
for key, data in reporter_dict.items():
    if key == "time_reporter":
        reporter_visualization.plot_time_reporter(data, plot=True)
    elif key == "species_reporter":
        reporter_visualization.plot_species_reporter(data, plot=True)
    elif key == "fitness_reporter":
        reporter_visualization.plot_fitness_reporter(data, plot=True)

# Plot genome
genome_visualization.draw_genome_graph(loaded_genome, draw_labels=False, plot=True)
text_visualization.print_genome(loaded_genome)

# Compare times
if compare_file is not None:
    _, compare_reporter_dict = file_save.load_genome_and_reporter(compare_file)
    compare_time_data = compare_reporter_dict["time_reporter"]
    selected_time_data = reporter_dict["time_reporter"]

    abs_all, factor_all, ratio_all = performance.speed_up_values_all(compare_time_data, selected_time_data,
                                                                     number_of_cores)
    abs_evaluation, factor_evaluation, ratio_evaluation = performance.speed_up_value_evaluation_time(compare_time_data,
                                                                                                     selected_time_data,
                                                                                                     number_of_cores)
    abs_compose, factor_compose, ratio_compose = performance.speed_up_value_compose_offspring(compare_time_data,
                                                                                              selected_time_data,
                                                                                              number_of_cores)
    abs_reproduction, factor_reproduction, ratio_reproduction = performance.speed_up_value_reproduction_time(
        selected_time_data, compare_time_data, number_of_cores)

    logger.info("Comparing evaluation times to genome in file: {}", compare_file)
    logger.info("Total Speedup: {}s, Ratio: {}, Efficiency: {}%", round(abs_all, 2), round(ratio_all, 2),
                round(ratio_all * 100, 2))
    logger.info("Evaluation Speedup: {}s, Ratio: {}, Efficiency: {}%", round(abs_evaluation, 2),
                round(factor_evaluation, 2), round(ratio_evaluation * 100, 2))
    logger.info("Reproduction Speedup: {}s, Ratio: {}, Efficiency: {}%", round(abs_compose, 2),
                round(factor_compose, 2), round(ratio_compose * 100, 2))
    logger.info("Compose OffSpring Speedup: {}s, Ratio: {}, Efficiency: {}%", round(abs_reproduction, 2),
                round(factor_reproduction, 2), round(ratio_reproduction * 100, 2))

for _ in range(amount_runs):
    func = challenge_dict[challenge]
    func(loaded_genome)

logger.info("Finished visualization")
