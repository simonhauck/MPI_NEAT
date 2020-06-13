from unittest import TestCase

from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.species import Species
from utils.reporter import species_reporter as sp


class FitnessEvaluationUtilsTest(TestCase):

    def setUp(self) -> None:
        self.agent1 = Agent(1, Genome(1, [], []))
        self.agent2 = Agent(2, Genome(1, [], []))
        self.agent3 = Agent(3, Genome(1, [], []))
        self.agent4 = Agent(4, Genome(1, [], []))
        self.agent5 = Agent(5, Genome(1, [], []))
        self.agent6 = Agent(6, Genome(1, [], []))
        self.agent7 = Agent(7, Genome(1, [], []))

        self.species_reporter = sp.SpeciesReporter()

    def test_add_generation_fitness_reporter(self):
        self.assertIsNone(self.species_reporter.min_generation)
        self.assertIsNone(self.species_reporter.max_generation)
        self.assertEqual({}, self.species_reporter.species_size_dict)

        species1 = Species(1, None,
                           [self.agent1, self.agent2, self.agent3, self.agent4, self.agent5, self.agent6, self.agent7])

        # Add first generation
        generation1 = Generation(3, 1, [], [species1])
        self.species_reporter = sp.add_generation_species_reporter(self.species_reporter, generation1)

        # Check after first generation
        self.assertEqual(3, self.species_reporter.min_generation)
        self.assertEqual(3, self.species_reporter.max_generation)
        self.assertEqual({1: ([3], [7])}, self.species_reporter.species_size_dict)

        # Add second generation with three species
        species1.members = [self.agent1, self.agent2, self.agent3]
        species2 = Species(2, None, [self.agent4, self.agent5, self.agent6])
        species3 = Species(3, None, [self.agent7])
        generation2 = Generation(4, 1, [], [species1, species2, species3])
        self.species_reporter = sp.add_generation_species_reporter(self.species_reporter, generation2)

        # Check after second generation
        self.assertEqual(3, self.species_reporter.min_generation)
        self.assertEqual(4, self.species_reporter.max_generation)
        self.assertEqual({1: ([3, 4], [7, 3]),
                          2: ([4], [3]),
                          3: ([4], [1])},
                         self.species_reporter.species_size_dict)

        # Add third generation with two species
        species3.members = [self.agent4, self.agent5, self.agent6, self.agent7]
        generation3 = Generation(5, 1, [], [species1, species3])
        self.species_reporter = sp.add_generation_species_reporter(self.species_reporter, generation3)

        # Check after third generation
        self.assertEqual(3, self.species_reporter.min_generation)
        self.assertEqual(5, self.species_reporter.max_generation)
        self.assertEqual({1: ([3, 4, 5], [7, 3, 3]),
                          2: ([4], [3]),
                          3: ([4, 5], [1, 4])},
                         self.species_reporter.species_size_dict)
