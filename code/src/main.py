import argparse
import sys
import time

import numpy as np
from loguru import logger

from examples.lunar_lander.lunar_lander import LunarLanderOptimizer
from examples.mountain_car.mountain_car import MountainCarOptimizer
from examples.pendulum.pendulum import PendulumOptimizer
from examples.pole_balancing.pole_balancing import PoleBalancingOptimizer
from examples.xor.xor_evaluation import XOROptimizer
from neat_single_core.neat_optimizer_single_core import NeatOptimizerSingleCore

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | {level}   | <level>{message}</level>",
           level="INFO")


def xor_single_evaluation(p_seed) -> int:
    xor_optimizer = XOROptimizer()
    xor_optimizer.evaluate(NeatOptimizerSingleCore(), p_seed)
    return xor_optimizer.solved_generation_number


def mountain_car_single_evaluation(p_seed) -> int:
    mountain_optimizer = MountainCarOptimizer()
    mountain_optimizer.evaluate(NeatOptimizerSingleCore(), p_seed)
    return 0


def pole_balancing_single_evaluation(p_seed) -> int:
    pole_balancing_optimizer = PoleBalancingOptimizer()
    pole_balancing_optimizer.evaluate(NeatOptimizerSingleCore(), p_seed)
    return 0


def pendulum_single_evaluation(p_seed) -> int:
    pendulum_optimizer = PendulumOptimizer()
    pendulum_optimizer.evaluate(NeatOptimizerSingleCore(), p_seed)
    return 0


def lunar_lander_single_evaluation(p_seed) -> int:
    lunar_lander_optimizer = LunarLanderOptimizer()
    lunar_lander_optimizer.evaluate(NeatOptimizerSingleCore(), p_seed)
    return 0


challenge_dict = {
    "xor": xor_single_evaluation,
    "mountain_car": mountain_car_single_evaluation,
    "pole_balancing": pole_balancing_single_evaluation,
    "pendulum": pendulum_single_evaluation,
    "lunar_lander": lunar_lander_single_evaluation
}

parser = argparse.ArgumentParser(description="Run the given NEAT examples")
parser.add_argument('challenge', type=str, help="Select one example challenge, that should be run",
                    choices=challenge_dict.keys())
parser.add_argument("-s", metavar="--seed", type=int, help="Seed for the evaluation")
parser.add_argument("-r", metavar="--repeat", type=int, default=1, help="Run the same challenge multiple times")

args = parser.parse_args()

challenge = args.challenge
seed = args.s
amount_runs = args.r

logger.info("Selected challenge: {}, Seed: {}, Amount Runs: {}".format(challenge, seed, amount_runs))

# Create random generator only if multiple runs should be done
rnd_generator = np.random.RandomState(seed) if amount_runs > 1 else None

statistics = {"evaluation_times": [], "generations": []}
for _ in range(amount_runs):
    evaluation_seed = seed if rnd_generator is None else rnd_generator.randint(2 ** 24)
    start_time = time.time()

    # Run challenge
    func = challenge_dict[challenge]
    generations = func(evaluation_seed)

    # Add statistics
    required_time = time.time() - start_time
    statistics["evaluation_times"].append(required_time)
    statistics["generations"].append(generations)

# Show final results
logger.info("-------- Program finished --------")
logger.info(
    "Min generations: {}, Max generations: {}, Mean: {}".
        format(min(statistics["generations"]), max(statistics["generations"]), np.mean(statistics["generations"])))
logger.info("Min Time: {}, Max Time: {}, Mean: {}".format(min(statistics["evaluation_times"]),
                                                          max(statistics["evaluation_times"]),
                                                          np.mean(statistics["evaluation_times"])))
