from typing import Dict

import gym
import numpy as np

from neat_core.optimizer.challenge import Challenge
from neural_network.neural_network_interface import NeuralNetworkInterface


class PendulumChallenge(Challenge):

    def __init__(self) -> None:
        self.env = None
        self.observation = None

    def initialization(self, **kwargs) -> None:
        self.env = gym.make("Pendulum-v0")

    def before_evaluation(self, **kwargs) -> None:
        # self.env.seed(1)
        self.observation = self.env.reset()

    def evaluate(self, neural_network: NeuralNetworkInterface, **kwargs) -> (float, Dict[str, object]):
        max_steps = 200

        solved_rounds = []
        fitness_values = []
        amount_runs = 10

        for _ in range(amount_runs):

            fitness = 0
            self.before_evaluation()

            for _ in range(max_steps):
                modified_obs = [self.observation[0], self.observation[1], self.observation[2] / 8]
                action = neural_network.activate(modified_obs)
                # Parse the result from range 0, 1 to range -2, 2
                force = -2 + action[0] * 4

                self.observation, reward, _, _ = self.env.step([force])

                if kwargs.get("show", False):
                    self.env.render()

                fitness += reward

            solved_rounds.append(fitness >= -200)
            fitness_values.append(fitness)

        return -(np.sum(fitness_values) ** 2), {"solved": all(solved_rounds), "fitness_values: ": fitness_values,
                                                "solved_rounds": solved_rounds}

    def clean_up(self, **kwargs):
        self.env.close()
