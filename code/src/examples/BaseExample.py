from abc import ABC, abstractmethod

from neat_core.models.genome import Genome
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.optimizer.neat_optimizer_callback import NeatOptimizerCallback
from utils.persistance import file_save


class BaseExample(NeatOptimizerCallback, ABC):

    @abstractmethod
    def evaluate(self, optimizer: NeatOptimizer, seed: int = None, **kwargs):
        """
        Evaluate the the challenge with the given optimizer
        :param optimizer: the optimizer that should be used
        :param seed: seed for the optimizer randomness
        :param kwargs: additional arguments
        :return: None
        """
        pass

    def visualize(self, genome_path: str, **kwargs) -> None:
        """
        Load a given genome from a file and visualize its performance
        :param genome_path: the path to the genome file
        :param kwargs: additional arguments
        :return: None
        """
        loaded_genome = file_save.load_genome_file(genome_path)
        self.visualize_genome(loaded_genome)

    def visualize_genome(self, genome: Genome, **kwargs) -> None:
        """
        Visualize the given genome and its performance
        :param genome: the genome that should be visualized
        :param kwargs: additioanl arguments
        :return: None
        """
        pass
