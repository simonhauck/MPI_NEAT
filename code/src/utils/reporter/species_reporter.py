from typing import Dict, List

from neat_core.models.generation import Generation


class SpeciesReporter(object):

    def __init__(self) -> None:
        self.min_generation: int = None
        self.max_generation: int = None
        # Key SpeciesID, Value: (GenerationsList, MemberSizeList)
        self.species_size_dict: Dict[int, (List[int], List[int])] = {}


def add_generation_species_reporter(reporter: SpeciesReporter, generation: Generation) -> SpeciesReporter:
    """
    Add the given generation to the given species reporter. The species reporter tracks, how many species exist in every
    species
    :param reporter: the species reporter, to which the data should be added
    :param generation: the generation with its species and members
    :return: the updated species reporter
    """
    # Set smallest/largest generation number
    reporter.min_generation = min(reporter.min_generation,
                                  generation.number) if reporter.min_generation is not None else generation.number
    reporter.max_generation = max(reporter.max_generation,
                                  generation.number) if reporter.max_generation is not None else generation.number

    for species in generation.species_list:
        if species.id_ in reporter.species_size_dict:
            # Update values
            generation_numbers, member_size = reporter.species_size_dict[species.id_]
            generation_numbers.append(generation.number)
            member_size.append(len(species.members))
        else:
            # Add a new entry
            generation_numbers = [generation.number]
            member_size = [len(species.members)]
            reporter.species_size_dict[species.id_] = (generation_numbers, member_size)

    return reporter
