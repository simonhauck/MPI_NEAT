from typing import Dict, List

from neat_core.models.generation import Generation
from neat_core.optimizer.neat_reporter import NeatReporter


class SpeciesReporterData(object):

    def __init__(self) -> None:
        self.min_generation: int = None
        self.max_generation: int = None
        # Key SpeciesID, Value: (GenerationsList, MemberSizeList)
        self.species_size_dict: Dict[int, (List[int], List[int])] = {}


class SpeciesReporter(NeatReporter):

    def __init__(self) -> None:
        self.data = SpeciesReporterData()

    def on_generation_evaluation_end(self, generation: Generation, reporters: List[NeatReporter]) -> None:
        """
        Add the given generation to the species reporter. The species reporter data tracks, how many species exist in
        every generation and how many members each species has
        :param generation: the generation that is added
        :param reporters: a list with all reporters
        :return:
        """
        self.data.min_generation = min(self.data.min_generation,
                                       generation.number) if self.data.min_generation is not None else generation.number
        self.data.max_generation = max(self.data.max_generation,
                                       generation.number) if self.data.max_generation is not None else generation.number

        for species in generation.species_list:
            if species.id_ in self.data.species_size_dict:
                # Update values
                generation_numbers, member_size = self.data.species_size_dict[species.id_]
                generation_numbers.append(generation.number)
                member_size.append(len(species.members))
            else:
                # Add a new entry
                generation_numbers = [generation.number]
                member_size = [len(species.members)]
                self.data.species_size_dict[species.id_] = (generation_numbers, member_size)

    def store_data(self) -> (bool, str, object):
        return True, "species_reporter", self.data
