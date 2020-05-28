import sys
from typing import Dict

import matplotlib.pyplot as plt
from loguru import logger

from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.agent import Agent
from neat_core.models.connection import Connection
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.challenge import Challenge
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.optimizer.neat_optimizer_callback import NeatOptimizerCallback
from neat_single_core.neat_optimizer_single_core import NeatOptimizerSingleCore
from neural_network.basic_neural_network import BasicNeuralNetwork
from neural_network.neural_network_interface import NeuralNetworkInterface
from utils.fitness_evaluation import fitness_evaluation_utils
from utils.visualization import generation_visualization
from utils.visualization import genome_visualization
from utils.visualization import text_visualization

logger.remove()
# logger.add(sys.stderr, level="DEBUG")
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | {level}   | <level>{message}</level>",
           level="INFO")


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

    def __init__(self) -> None:
        self.plot_data = generation_visualization.PlotData()

    def evaluate(self, optimizer: NeatOptimizer):
        optimizer.register_callback(self)
        config = NeatConfig()

        optimizer.evaluate(amount_input_nodes=2,
                           amount_output_nodes=1,
                           activation_function=modified_sigmoid_function,
                           challenge=ChallengeXOR(),
                           config=config,
                           seed=1)

    def evaluate_fix_structure(self, optimizer: NeatOptimizer):
        optimizer.register_callback(self)
        config = NeatConfig()

        genome = Genome(
            0, 0,
            [Node(0, NodeType.INPUT, 0, modified_sigmoid_function, 0),
             Node(1, NodeType.INPUT, 0, modified_sigmoid_function, 0),
             Node(2, NodeType.OUTPUT, 0, modified_sigmoid_function, 1),
             Node(3, NodeType.HIDDEN, 0, modified_sigmoid_function, 0.5),
             Node(4, NodeType.HIDDEN, 0, modified_sigmoid_function, 0.5)],
            [Connection(1, 0, 3, 0.1, True),
             Connection(2, 1, 3, 0.1, True),
             Connection(3, 0, 4, 0.1, True),
             Connection(4, 1, 4, 0.1, True),
             Connection(5, 3, 2, 0.1, True),
             Connection(6, 4, 2, 0.1, True)]
        )

        optimizer.evaluate_genome_structure(genome_structure=genome,
                                            challenge=ChallengeXOR(),
                                            config=config,
                                            seed=1)

    def on_initialization(self) -> None:
        logger.info("On initialization called...")

    def on_generation_evaluation_start(self, generation: Generation) -> None:
        logger.info(
            "Starting evaluation of generation {} with {} agents".format(generation.number, len(generation.agents)))

    def on_agent_evaluation_start(self, i: int, agent: Agent) -> None:
        logger.debug("Starting evaluation of agent {}...".format(i))

    def on_agent_evaluation_end(self, i: int, agent: Agent) -> None:
        logger.debug("Finished evaluation of agent {} with fitness {}".format(i, agent.fitness))

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        logger.info("Finished evaluation of generation {}".format(generation.number))
        self.plot_data.add_generation(generation)

    def on_cleanup(self) -> None:
        logger.info("Cleanup called...")

    def on_finish(self, generation: Generation) -> None:
        logger.info("OnFinish called with generation {}".format(generation.number))

        # Get the best agent
        agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        # Print genome
        text_visualization.print_agent(agent)

        # Draw genome
        plt.subplot(1, 2, 1)
        generation_visualization.plot_fitness_values(self.plot_data)
        plt.subplot(1, 2, 2)
        genome_visualization.draw_genome_graph(agent.genome, draw_labels=True)
        plt.show()

        # Print actual results
        nn = BasicNeuralNetwork()
        nn.build(agent.genome)
        for xor_tuple in ChallengeXOR.xor_tuples:
            actual_result = nn.activate([xor_tuple[0], xor_tuple[1]])
            logger.info("Input1: {}, Input2: {}, Expected Output: {}, Actual Output: {}".format(xor_tuple[0],
                                                                                                xor_tuple[1],
                                                                                                xor_tuple[2],
                                                                                                actual_result))

    def finish_evaluation(self, generation: Generation) -> bool:
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        return best_agent.additional_info["solved"]


if __name__ == '__main__':
    xor_optimizer = XOROptimizer()
    xor_optimizer.evaluate_fix_structure(NeatOptimizerSingleCore())
