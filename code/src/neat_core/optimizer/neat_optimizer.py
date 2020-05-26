from abc import ABC

from neat_core.models.genome import Genome
from neat_core.optimizer.challenge import Challenge
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer_callback import NeatOptimizerCallback


class NeatOptimizer(ABC):

    def __init__(self):
        self.callback: NeatOptimizerCallback = None

    def register_callback(self, callback: NeatOptimizerCallback) -> None:
        """
        Register the given callback to receive notifications
        :param callback: the callback that will be registered. Only one callback can be registered.
        :return: None
        """
        self.callback = callback

    def unregister_callback(self) -> None:
        """
        Remove the current registered callback (if one exists).
        :return: None
        """
        self.callback = None

    def evaluate(self,
                 amount_input_nodes: int,
                 amount_output_nodes: int,
                 activation_function,
                 challenge: Challenge,
                 config: NeatConfig,
                 seed: int) -> None:
        """
        Start the evaluation process with new genomes
        :param amount_input_nodes:  the number of inputs nodes in the genome
        :param amount_output_nodes: the number of output nodes in the genome
        :param activation_function: the used activation function in the input & output nodes
        :param challenge: the challenge, that should be solved by the genomes
        :param config: the NeatConfig with the required parameters
        :param seed: the seed for deterministic genomes
        :return: None
        """
        pass

    def evaluate_genome_structure(self, genome_structure: Genome, challenge: Challenge, config: NeatConfig, seed: int):
        """
        Create a generation with the given genome. Only the structure of the genome is used
        :param genome_structure: the blueprint for the genomes
        :param challenge: the challenge that should be solved
        :param config: a neat config that specifies the parameters
        :param seed: a seed for the challenge
        :return:
        """
        pass
