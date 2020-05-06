from abc import ABC

import numpy as np

from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.optimizer.challenge import Challenge
from neat_core.optimizer.neat_config import NeatConfig


class NeatOptimizer(ABC):

    def __init__(self):
        self.neat_config: NeatConfig = None
        self.generation: Generation = None
        self.challenge: Challenge = None

    def start_evaluation(self,
                         genome: Genome,
                         challenge: Challenge,
                         config: NeatConfig,
                         seed: int = None) -> None:
        self.neat_config = config
        self.challenge = challenge

        random_generator = np.random.RandomState(seed)

    def create_next_generation(self) -> Generation:
        pass

    def evaluate_generation(self) -> Generation:
        pass

    def stop_evaluation(self) -> None:
        pass
