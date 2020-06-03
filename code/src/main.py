import sys

from loguru import logger

from examples.mountain_car.mountain_car import MountainCarOptimizer
from examples.xor.xor_evaluation import XOROptimizer
from neat_single_core.neat_optimizer_single_core import NeatOptimizerSingleCore

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | {level}   | <level>{message}</level>",
           level="INFO")


def xor_single():
    xor_optimizer = XOROptimizer()
    xor_optimizer.evaluate(NeatOptimizerSingleCore())


def mountain_car_single():
    mountain_optimizer = MountainCarOptimizer()
    mountain_optimizer.evaluate(NeatOptimizerSingleCore())


challenge_dict = {
    "xor": xor_single,
    "mountain_car": mountain_car_single
}

if len(sys.argv) <= 1:
    logger.info("No arguments given -> Select a challenge to start the training process")
    logger.info("Possible Arguments:")
    for key in challenge_dict.keys():
        logger.info("Key: {}".format(key))
    exit(0)

param = sys.argv[1]
if param not in challenge_dict:
    logger.info("Argument not found")
    exit(-1)

run = challenge_dict[param]
run()
