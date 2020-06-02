import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
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
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | {level}   | <level>{message}</level>",
           level="INFO")


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


class XOROptimizer(NeatOptimizerCallback):

    def __init__(self) -> None:
        self.plot_data = generation_visualization.PlotData()

    def evaluate(self, optimizer: NeatOptimizer):
        optimizer.register_callback(self)
        config = NeatConfig(allow_recurrent_connections=False,
                            population_size=150,
                            compatibility_threshold=1.8,
                            connection_min_weight=-3,
                            connection_max_weight=3)

        seed = np.random.RandomState().randint(2 ** 24)
        # seed = 15545410
        # seed = 4931215
        # Good seed: 15545410
        # Generation 24: 11760111
        logger.info("Used Seed: {}".format(seed))

        optimizer.evaluate(amount_input_nodes=2,
                           amount_output_nodes=1,
                           activation_function=modified_sigmoid_function,
                           challenge=ChallengeXOR(),
                           config=config,
                           seed=seed)

    def evaluate_fix_structure(self, optimizer: NeatOptimizer):
        optimizer.register_callback(self)
        config = NeatConfig(allow_recurrent_connections=False,
                            probability_mutate_add_connection=0,
                            probability_mutate_add_node=0)

        genome = Genome(
            0,
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
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        self.plot_data.add_generation(generation)

        logger.info("Finished evaluation of generation {}".format(generation.number))
        logger.info("Best Fitness in Agent {}: {}".format(best_agent.id, best_agent.fitness))
        logger.info("Amount species: {}".format(len(generation.species_list)))

    def on_cleanup(self) -> None:
        logger.info("Cleanup called...")

    def on_finish(self, generation: Generation) -> None:
        logger.info("OnFinish called with generation {}".format(generation.number))

        # Get the best agent
        agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        # Print genome
        text_visualization.print_agent(agent)

        # Draw genome
        generation_visualization.plot_fitness_values(self.plot_data)
        plt.show()
        genome_visualization.draw_genome_graph(agent.genome, draw_labels=True)
        plt.show()

        # Print actual results
        challenge = ChallengeXOR()
        nn = BasicNeuralNetwork()
        nn.build(agent.genome)
        fitness, additional_info = challenge.evaluate(nn, show=True)
        logger.info(
            "Finished Evaluation - Fitness: {}, Challenge solved = {}".format(fitness, additional_info["solved"]))

    def finish_evaluation(self, generation: Generation) -> bool:
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        return best_agent.additional_info["solved"]


if __name__ == '__main__':
    xor_optimizer = XOROptimizer()
    # xor_optimizer.evaluate_fix_structure(NeatOptimizerSingleCore())
    xor_optimizer.evaluate(NeatOptimizerSingleCore())
