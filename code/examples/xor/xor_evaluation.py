from typing import Dict, List

from loguru import logger

from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.species import Species
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

    def __init__(self, optimizer: NeatOptimizer) -> None:
        self.optimizer = optimizer
        self.optimizer.register_callback(self)

        self.agent_solved = None

        self.optimizer.start_evaluation(amount_input_nodes=3,
                                        amount_output_nodes=1,
                                        activation_function=modified_sigmoid_function,
                                        challenge=ChallengeXOR(),
                                        config=NeatConfig())

    def on_initialization(self) -> None:
        logger.info("On initialization called...")

    def on_generation_evaluation_start(self, generation: Generation, species_list: List[Species]) -> None:
        logger.info(
            "Starting evaluation of generation {} with {} agents".format(generation.number, len(generation.agents)))

    def on_agent_evaluation_start(self, agent: Agent) -> None:
        logger.info("Starting evaluation of agent...")

    def on_agent_evaluation_end(self, agent: Agent) -> None:
        logger.info("Finished evaluation of agent with fitness {}".format(agent.fitness))

        if "solved" in agent.additional_info and agent.additional_info["solved"]:
            self.agent_solved = agent

    def on_generation_evaluation_end(self, generation: Generation, species_list: List[Species]) -> None:
        logger.info("Finished evaluation of generation {}".format(generation.number))

        if self.agent_solved is not None:
            self.optimizer.cleanup()
            logger.info("--------- Solution found ------------------")

    def on_cleanup(self) -> None:
        logger.info("Cleanup called...")


if __name__ == '__main__':
    xor_optimizer = XOROptimizer(NeatOptimizerSingleCore())
