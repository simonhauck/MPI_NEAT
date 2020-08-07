from typing import Dict

import gym
import numpy as np
from loguru import logger

from neat_core.optimizer.challenge import Challenge
from neural_network.neural_network_interface import NeuralNetworkInterface


class ChallengeMountainCar(Challenge):

    def __init__(self) -> None:
        self.env = None
        self.observation = None

    def initialization(self, **kwargs) -> None:
        self.env = gym.make("MountainCar-v0")

    def before_evaluation(self, **kwargs) -> None:
        self.env.seed(1111)
        self.observation = self.env.reset()

    def evaluate(self, neural_network: NeuralNetworkInterface, **kwargs) -> (float, Dict[str, object]):
        solved_rounds = []
        fitness_values = []
        max_x_values = []

        amount_runs = 10

        for _ in range(amount_runs):
            # Max episodes of environment, after which it terminates
            max_episodes = 200

            # Mix x position is -1.2, so that is never negative
            fitness = 1.3 + max_episodes

            # Reset environment
            self.before_evaluation()
            neural_network.reset()
            max_x_progress = self.observation[0]

            for _ in range(max_episodes):
                # Get action from neural network
                action = neural_network.activate(self.observation)
                index = np.argmax(action)

                self.observation, reward, done, info = self.env.step(index)

                # Render environment only if it is specifically requested
                if kwargs.get("show", False):
                    self.env.render()

                # Used for fitness
                fitness += reward
                if self.observation[0] > max_x_progress:
                    max_x_progress = self.observation[0]

                if done:
                    fitness += max_x_progress
                    break
            else:
                logger.error("Environment finished without done! Something is wrong..")

            if kwargs.get("show", False):
                logger.info("Finished run with Fitness: {}, Max X: {}".format(fitness, max_x_progress))

            # Add tracker information
            solved = fitness >= 90
            fitness_values.append(fitness)
            max_x_values.append(max_x_progress)
            solved_rounds.append(solved)

        return sum(fitness_values) ** 2, {"solved": all(solved_rounds),
                                          "max_x": max_x_values,
                                          "fitness_values": fitness_values,
                                          "solved_rounds": solved_rounds}

    def clean_up(self, **kwargs):
        self.env.close()
