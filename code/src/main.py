import argparse
import sys
import time

import numpy as np
from loguru import logger
from mpi4py import MPI

from examples.breakout.breakout import BreakoutOptimizer
from examples.lunar_lander.lunar_lander import LunarLanderOptimizer
from examples.mountain_car.mountain_car import MountainCarOptimizer
from examples.pendulum.pendulum import PendulumOptimizer
from examples.pole_balancing.pole_balancing import PoleBalancingOptimizer
from examples.xor.xor_evaluation import XOROptimizer
from neat_mpi.neat_optimizer_mpi import NeatOptimizerMPI
from neat_single_core.neat_optimizer_single_core import NeatOptimizerSingleCore

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | {level}   | <level>{message}</level>",
           level="INFO")


def xor_evaluation(p_seed, p_optimizer) -> int:
    xor_optimizer = XOROptimizer()
    xor_optimizer.evaluate(p_optimizer, p_seed)
    return 0 if xor_optimizer.solved_generation_number is None else xor_optimizer.solved_generation_number
    # return xor_optimizer.solved_generation_number


def mountain_car_evaluation(p_seed, p_optimizer) -> int:
    mountain_optimizer = MountainCarOptimizer()
    mountain_optimizer.evaluate(p_optimizer, p_seed)
    return 0


def pole_balancing_evaluation(p_seed, p_optimizer) -> int:
    pole_balancing_optimizer = PoleBalancingOptimizer()
    pole_balancing_optimizer.evaluate(p_optimizer, p_seed)
    return 0


def pendulum_evaluation(p_seed, p_optimizer) -> int:
    pendulum_optimizer = PendulumOptimizer()
    pendulum_optimizer.evaluate(p_optimizer, p_seed)
    return 0


def lunar_lander_evaluation(p_seed, p_optimizer) -> int:
    lunar_lander_optimizer = LunarLanderOptimizer()
    lunar_lander_optimizer.evaluate(p_optimizer, p_seed)
    return 0


def breakout_evaluation(p_seed, p_optimizer) -> int:
    breakout_optimizer = BreakoutOptimizer()
    breakout_optimizer.evaluate(p_optimizer, p_seed)
    return 0


def main_worker():
    global challenge, challenge_dict

    worker_func = challenge_dict[challenge]
    worker_func(None, NeatOptimizerMPI())


challenge_dict = {
    "xor": xor_evaluation,
    "mountain_car": mountain_car_evaluation,
    "pole_balancing": pole_balancing_evaluation,
    "pendulum": pendulum_evaluation,
    "lunar_lander": lunar_lander_evaluation,
    "breakout": breakout_evaluation
}

# TODO BUG
# Main.py is included twice with mpi worker process...
if sys.argv[0] == sys.argv[1]:
    del sys.argv[0]

# Add arguments
parser = argparse.ArgumentParser(description="Run the given NEAT examples")
parser.add_argument('challenge', type=str, help="Select one example challenge, that should be run",
                    choices=challenge_dict.keys())
parser.add_argument("-s", metavar="--seed", type=int, help="Seed for the evaluation")
parser.add_argument("-r", metavar="--repeat", type=int, default=1, help="Run the same challenge multiple times")
parser.add_argument("-o", metavar="--optimizer", type=str, default="single", choices=["single", "mpi"])

args = parser.parse_args()

challenge = args.challenge
seed = args.s
amount_runs = args.r
optimizer_type = args.o

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank != 0:
    main_worker()
    logger.info("Worker with rank {} completed main", rank)

if __name__ == '__main__':
    # Run specified challenge
    logger.info("Selected challenge: {}, Seed: {}, Amount Runs: {}".format(challenge, seed, amount_runs))

    # Create random generator only if multiple runs should be done
    rnd_generator = np.random.RandomState(seed) if amount_runs > 1 else None

    statistics = {"evaluation_times": [], "generations": []}
    for _ in range(amount_runs):

        evaluation_seed = seed if rnd_generator is None else rnd_generator.randint(2 ** 24)
        start_time = time.time()

        # Get the selected optimizer
        if optimizer_type == "single":
            optimizer = NeatOptimizerSingleCore()
        elif optimizer_type == "mpi":
            optimizer = NeatOptimizerMPI()

        # Run challenge
        func = challenge_dict[challenge]
        generations = func(evaluation_seed, optimizer)

        # Add statistics
        required_time = time.time() - start_time
        statistics["evaluation_times"].append(required_time)
        statistics["generations"].append(generations)

    # Show final results
    logger.info("-------- Program finished --------")
    logger.info("Provided seed:")
    logger.info(
        "Min generations: {}, Max generations: {}, Mean: {}".format(min(statistics["generations"]),
                                                                    max(statistics["generations"]),
                                                                    np.mean(statistics["generations"])))
    logger.info("Min Time: {}, Max Time: {}, Mean: {}".format(min(statistics["evaluation_times"]),
                                                              max(statistics["evaluation_times"]),
                                                              np.mean(statistics["evaluation_times"])))
