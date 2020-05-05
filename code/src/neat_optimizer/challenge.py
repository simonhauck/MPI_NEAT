from abc import ABC, abstractmethod
from typing import Dict

from neural_network.neural_network_interface import NeuralNetworkInterface


class Challenge(ABC):

    def initialization(self) -> None:
        """
        Initialize the challenge, only called once.
        :return: None
        """
        pass

    def before_evaluation(self) -> None:
        """
        Prepare the evaluation of a neural network. Called before every neural network.
        :return: None
        """
        pass

    @abstractmethod
    def evaluate(self, neural_network: NeuralNetworkInterface) -> (float, Dict[str, object]):
        """
        Evaluate a neural network.
        :param neural_network: the neural network, that should be evaluated
        :return: float - the calculated fitness value, an optionally dictionary with additional info
        """
        pass

    def after_evaluation(self) -> None:
        """
        Finish the evaluation. Called always after the evaluation finishes.
        :return: None
        """
        pass

    def clean_up(self):
        """
        Cleanup, when the evaluation of all networks is finished.
        :return: None
        """
        pass
