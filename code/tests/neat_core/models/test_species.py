from unittest import TestCase

from neat_core.activation_function import modified_sigmoid_activation
from neat_core.models.agent import Agent
from neat_core.models.species import Species
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service.generation_service import create_genome_structure
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class SpeciesTest(TestCase):

    def test_species(self):
        genome1 = create_genome_structure(5, 2, modified_sigmoid_activation, NeatConfig(),
                                          InnovationNumberGeneratorSingleCore())

        genome2 = create_genome_structure(5, 2, modified_sigmoid_activation, NeatConfig(),
                                          InnovationNumberGeneratorSingleCore())
        members = [Agent(1, genome1)]

        species = Species(1, genome2, members, max_species_fitness=10, generation_max_species_fitness=3,
                          adjust_fitness=8)

        self.assertEqual(genome2, species.representative)
        self.assertEqual(8, species.adjusted_fitness)
        self.assertEqual(10, species.max_species_fitness)
        self.assertEqual(3, species.generation_max_species_fitness)
        self.assertEqual(members, species.members)
        self.assertEqual(1, species.id_)
