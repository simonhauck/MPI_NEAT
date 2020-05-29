from typing import List

from neat_core.models.agent import Agent
from neat_core.models.genome import Genome


class Species(object):

    def __init__(self, id_: int, representative: Genome, members: List[Agent], max_species_fitness: float = None,
                 generation_max_species_fitness: float = None, adjust_fitness: float = None) -> None:
        self.id_ = id_
        self.representative: Genome = representative
        self.adjusted_fitness: float = adjust_fitness

        self.max_species_fitness = max_species_fitness
        self.generation_max_species_fitness: float = generation_max_species_fitness

        self.members: List[Agent] = members
