from unittest import TestCase

from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.species import Species
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service.generation_service import create_genome_structure
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class GenerationTest(TestCase):

    def test_generation(self):
        genome1 = create_genome_structure(5, 2, modified_sigmoid_function, NeatConfig(),
                                          InnovationNumberGeneratorSingleCore())

        genome2 = create_genome_structure(5, 2, modified_sigmoid_function, NeatConfig(),
                                          InnovationNumberGeneratorSingleCore())

        agent1 = Agent(genome1)
        agent2 = Agent(genome2)
        members = [agent1, agent2]

        species = [Species(genome2, [agent1, agent2], max_species_fitness=10, generation_max_species_fitness=3,
                           sum_fitness=12, sum_adjusted_fitness=8)]

        generation = Generation(2, members, species)

        self.assertEqual(2, generation.number)
        self.assertEqual(species, generation.species_list)
        self.assertEqual(members, generation.agents)
