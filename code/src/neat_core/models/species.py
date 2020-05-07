from typing import List

from neat_core.models.agent import Agent
from neat_core.models.genome import Genome


class Species(object):

    def __init__(self, representative: Genome, members: List[Agent], max_species_fitness: float = None,
                 generation_max_species_fitness: float = None, sum_fitness: float = None,
                 sum_adjusted_fitness: float = None) -> None:
        self.representative: Genome = representative
        self.sum_fitness: float = sum_fitness
        self.sum_adjusted_fitness: float = sum_adjusted_fitness

        self.max_species_fitness = max_species_fitness
        self.generation_max_species_fitness: float = generation_max_species_fitness

        self.members: List[Agent] = members
