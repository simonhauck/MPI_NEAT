from typing import Dict

import gym
import numpy as np
from gym import wrappers
from loguru import logger
from pyvirtualdisplay import Display

from neat_core.optimizer.challenge import Challenge
from neural_network.neural_network_interface import NeuralNetworkInterface


class ChallengeMountainCar(Challenge):

    def __init__(self) -> None:
        self.env = None
        self.observation = None
        self.virtual_display = None

    def initialization(self, **kwargs) -> None:
        self.env = gym.make("MountainCar-v0")
        if "show" in kwargs:
            self.virtual_display = Display(visible=0, size=(1400, 900))
            self.virtual_display.start()
            self.env = gym.wrappers.Monitor(self.env, "/tmp/mountain_car")

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
            #    logger.info("Observation: {}, Selected Action: {}, Raw NN: {}".format(self.observation, index, action))

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

        if "show" in kwargs:
            logger.info("Closing virtual display")
            self.virtual_display.stop()
