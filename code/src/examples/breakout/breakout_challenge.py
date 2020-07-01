from typing import Dict

import gym
import numpy as np

from neat_core.optimizer.challenge import Challenge
from neural_network.neural_network_interface import NeuralNetworkInterface


class BreakoutChallenge(Challenge):

    def __init__(self) -> None:
        self.env = None
        self.observation = None

    def initialization(self, **kwargs) -> None:
        self.env = gym.make('Breakout-ramNoFrameskip-v0')

    def before_evaluation(self, **kwargs) -> None:
        self.env.seed(0)
        self.observation = self.env.reset()

    def evaluate(self, neural_network: NeuralNetworkInterface, **kwargs) -> (float, Dict[str, object]):
        fitness = 1

        last_lives = 5
        should_fire = True

        for _ in range(5000):
            if should_fire:
                # Fire the ball
                self.env.step(1)
                should_fire = False

            normalized = [v / 256.0 for v in self.observation]
            action = neural_network.activate(normalized)
            index = np.argmax(action)

            self.observation, reward, done, info = self.env.step(2 + index)

            # self.env.render()

            if info["ale.lives"] != last_lives:
                should_fire = True
                last_lives = info["ale.lives"]

            # Render environment only if it is specifically requested
            if kwargs.get("record", False) or kwargs.get("show", False):
                self.env.render()

            # Used for fitness
            fitness += reward

            if done:
                break

        return fitness, {"solved": fitness >= 30}

    def clean_up(self, **kwargs):
        self.env.close()
