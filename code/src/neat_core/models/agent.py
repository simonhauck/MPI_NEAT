from neat_core.models.genome import Genome
from neural_network.neural_network_interface import NeuralNetworkInterface


class Agent(object):

    def __init__(self, genome: Genome) -> None:
        self.genome: Genome = genome
        self.neural_network: NeuralNetworkInterface = None

        # Fitness value
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0
