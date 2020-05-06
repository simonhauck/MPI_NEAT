from unittest import TestCase

from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class InnovationNumberGeneratorSingleCoreTest(TestCase):

    def setUp(self) -> None:
        self.generator = InnovationNumberGeneratorSingleCore()

    def test(self):
        self.assertEqual(0, self.generator.get_connection_innovation_number())
        self.assertEqual(1, self.generator.get_connection_innovation_number())
        self.assertEqual(2, self.generator.get_connection_innovation_number())
        self.assertEqual(3, self.generator.get_connection_innovation_number())

        self.assertEqual(0, self.generator.get_node_innovation_number())
        self.assertEqual(1, self.generator.get_node_innovation_number())
        self.assertEqual(2, self.generator.get_node_innovation_number())
        self.assertEqual(3, self.generator.get_node_innovation_number())
