from typing import Dict

from loguru import logger

from neat_core.optimizer.challenge import Challenge
from neural_network.neural_network_interface import NeuralNetworkInterface


class ChallengeXOR(Challenge):
    xor_tuples = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0]
    ]

    def evaluate(self, neural_network: NeuralNetworkInterface, **kwargs) -> (float, Dict[str, object]):
        fitness_val = 4.0
        solved = True

        for xor_input in ChallengeXOR.xor_tuples:
            # Calculate xor with value
            inputs = [xor_input[0], xor_input[1]]
            result_array = neural_network.activate(inputs)
            result = result_array[0]

            # result = (1 + result) / 2

            # Print results, if flag is given
            if "show" in kwargs:
                logger.info(
                    "Activate neural net - Inputs: {}, Expected Output: {}, Output: {}".format(inputs, xor_input[2],
                                                                                               result_array))

            # Remove difference from fitness
            difference = (abs(xor_input[2] - result))
            fitness_val -= difference

            # Challenge is solved, if the difference is smaller than 0.5 for every input xor_input
            solved = solved and difference < 0.5

        # Remaining fitness value is squared
        fitness_val = fitness_val ** 2

        return fitness_val, {"solved": solved}
