from typing import Dict

import gym
import numpy as np
from gym import wrappers
from loguru import logger
from pyvirtualdisplay import Display

from neat_core.optimizer.challenge import Challenge
from neural_network.neural_network_interface import NeuralNetworkInterface


class PoleBalancingChallenge(Challenge):

    def __init__(self) -> None:
        # https://github.com/openai/gym/wiki/CartPole-v0
        self.env = None
        self.observation = None
        self.virtual_display = None

    def initialization(self, **kwargs) -> None:
        self.env = gym.make("CartPole-v1")

        if kwargs.get("record", False):
            self.virtual_display = Display(visible=0, size=(1400, 900))
            self.virtual_display.start()
            self.env = gym.wrappers.Monitor(self.env, "/tmp/pole_balancing", video_callable=lambda episode_id: True)

    def before_evaluation(self, **kwargs) -> None:
        # self.env.seed(0)
        self.observation = self.env.reset()

    def evaluate(self, neural_network: NeuralNetworkInterface, **kwargs) -> (float, Dict[str, object]):
        max_episodes = 500
        runs = 10
        fitness = 0

        for _ in range(runs):
            self.before_evaluation()
            done = False
            while not done:
                action = neural_network.activate(self.observation)
                index = np.argmax(action)

                # old_observation = self.observation

                self.observation, reward, done, info = self.env.step(index)

                # Render environment only if it is specifically requested
                if kwargs.get("record", False) or kwargs.get("show", False):
                    self.env.render()

                # if kwargs.get("show", False):
                #     logger.info("Observation {}, Action {}".format(old_observation, index))

                fitness += reward

        return fitness ** 2, {"solved": fitness >= max_episodes * runs, "steps": fitness}

    def clean_up(self, **kwargs):
        self.env.close()

        if kwargs.get("record", False):
            logger.info("Closing virtual display")
            self.virtual_display.stop()
