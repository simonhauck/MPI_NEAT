from typing import List

from neat_core.models.species import Species
from neat_core.models.species_id_generator import SpeciesIDGeneratorInterface


class SpeciesIDGeneratorSingleCore(SpeciesIDGeneratorInterface):

    def __init__(self, existing_species: List[Species] = None) -> None:
        if existing_species is None or len(existing_species) == 0:
            self.next_id = 0
        else:
            self.next_id = max([s.id_ for s in existing_species]) + 1

    def get_species_id(self) -> int:
        tmp = self.next_id
        self.next_id += 1
        return tmp
