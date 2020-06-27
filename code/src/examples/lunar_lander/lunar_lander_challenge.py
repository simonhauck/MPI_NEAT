from typing import Dict

import gym
import numpy as np

from neat_core.optimizer.challenge import Challenge
from neural_network.neural_network_interface import NeuralNetworkInterface


class LunarLanderChallenge(Challenge):

    def __init__(self) -> None:
        self.env = None
        self.observation = None

    def initialization(self, **kwargs) -> None:
        self.env = gym.make("LunarLander-v2")

    def before_evaluation(self, **kwargs) -> None:
        # self.env.seed(0)
        self.observation = self.env.reset()

    def evaluate(self, neural_network: NeuralNetworkInterface, **kwargs) -> (float, Dict[str, object]):
        solved_rounds = []
        fitness_values = []
        amount_runs = 3

        for _ in range(amount_runs):
            self.before_evaluation()
            done = False
            fitness = 0
            while not done:
                action = neural_network.activate(self.observation)
                index = np.argmax(action)

                self.observation, reward, done, info = self.env.step(index)

                # Render environment only if it is specifically requested
                if kwargs.get("record", False) or kwargs.get("show", False):
                    self.env.render()

                # Used for fitness
                fitness += reward

            fitness_values.append(fitness)
            solved_rounds.append(fitness >= 200)

        return sum(fitness_values), {"solved": all(solved_rounds), "fitness_values": fitness_values,
                                     "solved_rounds": solved_rounds}

    def clean_up(self, **kwargs):
        self.env.close()
