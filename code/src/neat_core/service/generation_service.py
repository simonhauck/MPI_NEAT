from typing import List

from neat_core.models.genome import Genome
from neat_core.models.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.optimizer import NeatConfig


def create_initial_genomes(input_neurons: int, output_neurons: int, generator: InnovationNumberGeneratorInterface,
                           config: NeatConfig, seed: int = None) -> List[Genome]:
    return []
