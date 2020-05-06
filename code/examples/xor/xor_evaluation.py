from typing import Dict

from loguru import logger

from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.challenge import Challenge
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.optimizer.neat_optimizer_callback import NeatOptimizerCallback
from neat_single_core.neat_optimizer_single_core import NeatOptimizerSingleCore
from neural_network.neural_network_interface import NeuralNetworkInterface


class ChallengeXOR(Challenge):
    xor_tuples = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0]
    ]

    def evaluate(self, neural_network: NeuralNetworkInterface) -> (float, Dict[str, object]):
        fitness_val = 4.0
        solved = True

        for xor_input in ChallengeXOR.xor_tuples:
            # Calculate xor with value
            result_array = neural_network.activate([xor_input[0], xor_input[1]])
            result = result_array[0]

            # Remove difference from fitness
            difference = (abs(xor_input[2] - result))
            fitness_val -= difference

            # Challenge is solved, if the difference is smaller than 0.5 for every input xor_input
            solved = solved and difference < 0.5

        # Remaining fitness value is squared
        fitness_val = fitness_val ** 2

        return fitness_val, {"solved": solved}


class XOROptimizer(NeatOptimizerCallback):

    def __init__(self, optimizer: NeatOptimizer, seed: int = None) -> None:
        logger.info("Start running XOR example...")
        self.optimizer = optimizer

        logger.debug("Creating start genome...")
        start_genome = Genome(id_=0, seed=0,
                              nodes=[
                                  Node(innovation_number=1,
                                       type_=NodeType.INPUT,
                                       activation_function=modified_sigmoid_function,
                                       x_position=0),
                                  Node(innovation_number=2,
                                       type_=NodeType.INPUT,
                                       activation_function=modified_sigmoid_function,
                                       x_position=0),
                                  Node(innovation_number=3,
                                       type_=NodeType.OUTPUT,
                                       activation_function=modified_sigmoid_function,
                                       x_position=1)
                              ],
                              connections=[
                                  Connection(innovation_number=4,
                                             input_node=1,
                                             output_node=3,
                                             weight=0,
                                             enabled=True),
                                  Connection(innovation_number=5,
                                             input_node=2,
                                             output_node=3,
                                             weight=0,
                                             enabled=True)
                              ]
                              )

        logger.debug("Starting evaluation...")
        self.optimizer.start_evaluation(start_genome, ChallengeXOR(), NeatConfig(), seed)

    def on_initialization(self):
        logger.info("Called on_initialization for the XOR challenge...")

    def on_generation_evaluation_start(self):
        logger.info("Called on_generation_evaluation_start for the XOR challenge...")

    def on_agent_evaluation_start(self):
        logger.trace("Called on_agent_evaluation_start for the XOR challenge...")

    def on_agent_evaluation_end(self):
        logger.trace("Called on_agent_evaluation_end for the XOR challenge...")

    def on_generation_evaluation_end(self):
        logger.info("Called on_generation_evaluation_start for the XOR challenge...")

        challenge_solved = True
        if challenge_solved:
            self.optimizer.stop_evaluation()

    def on_evaluation_stopped(self) -> None:
        logger.info("Called on_evaluation_stopped for XOR challenge...")


if __name__ == '__main__':
    xor_optimizer = XOROptimizer(NeatOptimizerSingleCore(), 1)
