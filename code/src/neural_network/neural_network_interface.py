from typing import List

from neat_core.models.genome import Genome


class NeuralNetworkInterface(object):

    def build(self, genome: Genome) -> None:
        """
        Initialize the neural network at the beginning. If the network is not initialized, it must not function properly
        :return: None
        """
        pass

    def reset(self) -> None:
        """
        Reset the network to the initial state.
        :return: None
        """
        pass

    def activate(self, inputs: List[float]) -> List[float]:
        """
        Activate the neural network with the given inputs. The network calculates the result and returns them.
        :param inputs: the inputs for the neural network. The size must match the input neurons
        :return: the calculates result of the neural network
        """
        pass
