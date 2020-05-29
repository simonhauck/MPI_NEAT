from typing import Dict

from neat_core.models.genome import Genome
from neural_network.neural_network_interface import NeuralNetworkInterface


class Agent(object):

    def __init__(self, id_: int, genome: Genome) -> None:
        self.id = id_
        self.genome: Genome = genome
        self.neural_network: NeuralNetworkInterface = None

        # TODO remove adjusted fitness
        # Fitness value
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0

        # Challenge info
        self.additional_info: Dict[str, object] = {}
