import argparse
import sys

from loguru import logger

from examples.mountain_car.mountain_car import MountainCarOptimizer
from examples.pole_balancing.pole_balancing import PoleBalancingOptimizer
from examples.xor.xor_evaluation import XOROptimizer

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | {level}   | <level>{message}</level>",
           level="INFO")


def xor_visualization(genome_path: str):
    xor_optimizer = XOROptimizer()
    xor_optimizer.visualize(genome_path)
    return xor_optimizer.solved_generation_number


def mountain_car_single_visualization(genome_path: str):
    mountain_optimizer = MountainCarOptimizer()
    mountain_optimizer.visualize(genome_path)
    return 0


def pole_balancing_single_visualization(genome_path: str):
    pole_balancing_optimizer = PoleBalancingOptimizer()
    pole_balancing_optimizer.visualize(genome_path)
    return 0


challenge_dict = {
    "xor": xor_visualization,
    "mountain_car": mountain_car_single_visualization,
    "pole_balancing": pole_balancing_single_visualization
}

parser = argparse.ArgumentParser(description="Visualize stored NEAT genomes in the challenges")
parser.add_argument('challenge', type=str, help="Select one example challenge, that should be run",
                    choices=challenge_dict.keys())
parser.add_argument("genome_file", type=str, help="Path to the genome that should be loaded")
parser.add_argument("-r", metavar="--repeat", type=int, default=1, help="Run the same challenge multiple times")

args = parser.parse_args()

challenge = args.challenge
genome_file = args.genome_file
amount_runs = args.r

logger.info("Selected challenge: {}, Amount Runs: {}".format(challenge, amount_runs))

for _ in range(amount_runs):
    func = challenge_dict[challenge]
    func(genome_file)

logger.info("Finished visualization")
