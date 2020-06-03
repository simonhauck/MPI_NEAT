import os.path as path
import site
import sys

parentdir = path.abspath(path.join(__file__, "../../.."))
site.addsitedir(parentdir + "/src/")
print(sys.path)

from typing import Dict
from loguru import logger

import gym
import matplotlib.pyplot as plt
import numpy as np
import progressbar

from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
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


class ChallengeMountainCar(Challenge):

    def __init__(self) -> None:
        self.env = None
        self.observation = None

    def initialization(self, **kwargs) -> None:
        self.env = gym.make("MountainCar-v0")

    def before_evaluation(self, **kwargs) -> None:
        self.observation = self.env.reset()

    def evaluate(self, neural_network: NeuralNetworkInterface, **kwargs) -> (float, Dict[str, object]):
        # Max episodes of environment, after which it terminates
        max_episodes = 200

        # Mix x position is -1.2, so that is never negative
        fitness = 1.3 + max_episodes
        max_x_progress = self.observation[0]

        for _ in range(max_episodes):
            # Get action from neural network
            action = neural_network.activate(self.observation)
            index = np.argmax(action)

            # if "show" in kwargs:
            #     logger.info("Observation: {}, Selected Action: {}, Raw NN: {}".format(self.observation, index, action))

            self.observation, reward, done, info = self.env.step(index)

            if "show" in kwargs:
                self.env.render()

            # Used for fitness
            fitness += reward
            if self.observation[0] > max_x_progress:
                max_x_progress = self.observation[0]

            if done:
                break
        else:
            logger.error("Environment finished without done! Something is wrong..")

        solved = max_x_progress >= 0.5
        return (fitness + max_x_progress) ** 2, {"solved": solved, "max_x": max_x_progress}

    def clean_up(self, **kwargs):
        self.env.close()


class MountainOptimizer(NeatOptimizerCallback):

    def __init__(self) -> None:
        self.plot_data = generation_visualization.PlotData()
        self.challenge = None

        self.progressbar = None
        self.progressbar_widgets = [
            'Generation: ', progressbar.Percentage(),
            ' ', progressbar.Bar(marker='#', left='[', right=']'),
            ' ', progressbar.Counter('Evaluated Agents: %(value)03d'),
            ', ', progressbar.ETA()
        ]
        self.progressbar_max = 0

    def evaluate(self, optimizer: NeatOptimizer):
        # Good seed: 11357659, pop 400, threshold 1.0, min max weight = -3,3

        optimizer.register_callback(self)
        config = NeatConfig(population_size=400,
                            compatibility_threshold=1.0,
                            connection_min_weight=-3,
                            connection_max_weight=3)

        self.progressbar_max = config.population_size

        seed = np.random.RandomState().randint(2 ** 24)
        logger.info("Used Seed: {}".format(seed))

        self.challenge = ChallengeMountainCar()

        optimizer.evaluate(amount_input_nodes=2, amount_output_nodes=3, activation_function=modified_sigmoid_function,
                           challenge=self.challenge, config=config, seed=seed)

    def on_initialization(self) -> None:
        logger.info("On initialization called...")

    def on_generation_evaluation_start(self, generation: Generation) -> None:
        logger.info(
            "Starting evaluation of generation {} with {} agents".format(generation.number, len(generation.agents)))
        self.progressbar = progressbar.ProgressBar(widgets=self.progressbar_widgets, max_value=self.progressbar_max)
        self.progressbar.start()

    def on_agent_evaluation_start(self, i: int, agent: Agent) -> None:
        logger.debug("Starting evaluation of agent {}...".format(i))

    def on_agent_evaluation_end(self, i: int, agent: Agent) -> None:
        logger.debug("Finished evaluation of agent {} with fitness {}".format(i, agent.fitness))
        self.progressbar.update(i + 1)

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        self.progressbar.finish()
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        self.plot_data.add_generation(generation)

        logger.info("Finished evaluation of generation {}".format(generation.number))
        logger.info("Best Fitness in Agent {}: {}, AdditionalInfo: {}".format(best_agent.id, best_agent.fitness,
                                                                              best_agent.additional_info))
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

        # Run best genome in endless loop
        nn = BasicNeuralNetwork()
        nn.build(agent.genome)

        challenge = ChallengeMountainCar()
        challenge.initialization()

        while True:
            challenge.before_evaluation()
            fitness, additional_info = challenge.evaluate(nn, show=False)
            logger.info("Finished Neural network  Fitness: {}, Info: {}".format(fitness, additional_info))

    def finish_evaluation(self, generation: Generation) -> bool:
        # best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        # return best_agent.additional_info["solved"]
        return generation.number >= 40


if __name__ == '__main__':
    mountain_optimizer = MountainOptimizer()
    mountain_optimizer.evaluate(NeatOptimizerSingleCore())
