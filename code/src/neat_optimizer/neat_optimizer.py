from abc import ABC

import numpy as np

from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_optimizer.neat_config import NeatConfig


class NeatOptimizer(ABC):

    def __init__(self):
        self.neat_config: NeatConfig = None
        self.generation: Generation = None

    def create_initial_generation(self, genome: Genome, config: NeatConfig, seed=None) -> None:
        self.neat_config = config

        random_generator = np.random.RandomState(seed)

    def create_next_generation(self) -> Generation:
        pass

    def evaluate_generation(self):
        pass
