from unittest import TestCase

from neat_core.activation_function import modified_sigmoid_activation
from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.species import Species
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.service.generation_service import create_genome_structure
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class GenerationTest(TestCase):

    def test_generation(self):
        genome1 = create_genome_structure(5, 2, modified_sigmoid_activation, NeatConfig(),
                                          InnovationNumberGeneratorSingleCore())

        genome2 = create_genome_structure(5, 2, modified_sigmoid_activation, NeatConfig(),
                                          InnovationNumberGeneratorSingleCore())

        agent1 = Agent(1, genome1)
        agent2 = Agent(2, genome2)
        members = [agent1, agent2]

        species = [Species(1, genome2, [agent1, agent2], max_species_fitness=10, generation_max_species_fitness=3,
                           adjust_fitness=8)]

        seed = 10
        generation = Generation(2, seed, members, species)

        self.assertEqual(2, generation.number)
        self.assertEqual(species, generation.species_list)
        self.assertEqual(members, generation.agents)
        self.assertEqual(seed, generation.seed)
