from unittest import TestCase

from neat_core.models.species import Species
from neat_single_core.species_id_generator_single_core import SpeciesIDGeneratorSingleCore


class InnovationNumberGeneratorSingleCoreTest(TestCase):

    def setUp(self) -> None:
        self.generator = SpeciesIDGeneratorSingleCore()

    def test_get_species_id(self):
        self.assertEqual(0, self.generator.get_species_id())
        self.assertEqual(1, self.generator.get_species_id())

    def test_get_species_id2(self):
        generator = SpeciesIDGeneratorSingleCore([Species(4, None, None), Species(6, None, None)])
        self.assertEqual(7, generator.get_species_id())
        self.assertEqual(8, generator.get_species_id())
