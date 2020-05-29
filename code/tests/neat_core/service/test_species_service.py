from unittest import TestCase

import neat_core.service.generation_service as gs
import neat_core.service.reproduction_service as rs
import neat_core.service.species_service as ss
from neat_core.activation_function import step_function
from neat_core.models.agent import Agent
from neat_core.models.connection import Connection
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.models.species import Species
from neat_core.optimizer.neat_config import NeatConfig
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore
from neat_single_core.species_id_generator_single_core import SpeciesIDGeneratorSingleCore


class SpeciesServiceTest(TestCase):

    def setUp(self) -> None:
        self.config = NeatConfig(compatibility_factor_matching_genes=1, compatibility_factor_disjoint_genes=2,
                                 compatibility_threshold=2.5, population_size=8)

        self.g1_nodes = [
            Node(1, NodeType.INPUT, 0, step_function, 0),
            Node(2, NodeType.INPUT, 0, step_function, 0),
            Node(3, NodeType.OUTPUT, 1.2, step_function, 1),
            Node(4, NodeType.HIDDEN, 1.5, step_function, 0.5),
            Node(6, NodeType.HIDDEN, 0.5, step_function, 0.5),
            Node(7, NodeType.HIDDEN, 0.2, step_function, 0.25)
        ]
        self.g1_connections = [
            Connection(1, 1, 3, 1.2, True),
            Connection(2, 2, 3, 0.5, False),
            Connection(3, 1, 4, -1.2, True),
            Connection(4, 4, 3, 0.2, True),
            Connection(5, 2, 6, 2.0, True),
            Connection(6, 6, 3, -1.1, False)
        ]
        self.g1 = Genome(1, self.g1_nodes, self.g1_connections)

        self.g2_nodes = [
            Node(1, NodeType.INPUT, 0, step_function, 0),
            Node(2, NodeType.INPUT, 0, step_function, 0),
            Node(3, NodeType.OUTPUT, 0.2, step_function, 1),
            Node(4, NodeType.HIDDEN, 1.2, step_function, 0.5),
            Node(5, NodeType.HIDDEN, 2.8, step_function, 0.5)
        ]

        self.g2_connections = [
            Connection(1, 1, 3, 0.8, True),
            Connection(2, 2, 3, 1.5, True),
            Connection(3, 1, 4, 1.2, True),
            Connection(4, 4, 3, 3.2, True),
            Connection(6, 6, 3, -1.1, False),
            Connection(7, 6, 3, -0.1, False),
            Connection(8, 1, 4, -1.1, False)
        ]
        self.g2 = Genome(2, self.g2_nodes, self.g2_connections)

        self.agent1 = Agent(1, self.g1)
        self.agent1.fitness = 1
        self.agent2 = Agent(2, self.g2)
        self.agent2.fitness = 2

        # Add some more agents, and complete species
        self.inno_num_generator = InnovationNumberGeneratorSingleCore()
        self.species_id_generator = SpeciesIDGeneratorSingleCore()
        self.genome1 = gs.create_genome_structure(2, 1, step_function, self.config, self.inno_num_generator)
        self.genome2 = gs.create_genome_structure(2, 1, step_function, self.config, self.inno_num_generator)
        self.genome3 = gs.create_genome_structure(2, 1, step_function, self.config, self.inno_num_generator)
        self.genome4 = gs.create_genome_structure(2, 1, step_function, self.config, self.inno_num_generator)
        self.genome5 = gs.create_genome_structure(2, 1, step_function, self.config, self.inno_num_generator)
        self.genome6 = gs.create_genome_structure(2, 1, step_function, self.config, self.inno_num_generator)
        self.genome7 = gs.create_genome_structure(2, 1, step_function, self.config, self.inno_num_generator)
        self.genome8 = gs.create_genome_structure(2, 1, step_function, self.config, self.inno_num_generator)

        self.agent3 = Agent(3, self.genome3)
        self.agent3.fitness = 3
        self.agent4 = Agent(4, self.genome4)
        self.agent4.fitness = 4
        self.agent5 = Agent(5, self.genome5)
        self.agent5.fitness = 5
        self.agent6 = Agent(6, self.genome6)
        self.agent6.fitness = 6
        self.agent7 = Agent(7, self.genome7)
        self.agent7.fitness = 7
        self.agent8 = Agent(8, self.genome8)
        self.agent8.fitness = 8

        self.species1 = Species(self.species_id_generator.get_species_id(), self.agent1.genome,
                                [self.agent1, self.agent2, self.agent3], max_species_fitness=1.5,
                                generation_max_species_fitness=10)
        self.species2 = Species(self.species_id_generator.get_species_id(), self.agent4.genome,
                                [self.agent4, self.agent5, self.agent6], max_species_fitness=7,
                                generation_max_species_fitness=5)
        self.species3 = Species(self.species_id_generator.get_species_id(), self.agent6.genome,
                                [self.agent7, self.agent8], max_species_fitness=0, generation_max_species_fitness=6)

        self.generation = Generation(21, 2,
                                     agents=[self.agent1, self.agent2, self.agent3, self.agent4, self.agent5,
                                             self.agent6, self.agent7, self.agent8],
                                     species_list=[self.species1, self.species2, self.species3])

    def test_calculate_genetic_distance(self):
        genetic_distance = ss.calculate_genetic_distance(self.g1, self.g2, self.config)
        genetic_distance2 = ss.calculate_genetic_distance(self.g2, self.g1, self.config)

        disjoint_gene_value_node = (3 / 6) * self.config.compatibility_factor_disjoint_genes
        matching_genes_value_node = 0.325 * self.config.compatibility_factor_matching_genes
        disjoint_gene_value_connection = (3 / 7) * self.config.compatibility_factor_disjoint_genes
        matching_genes_value_connection = 1.36 * self.config.compatibility_factor_matching_genes

        self.assertEqual(genetic_distance, genetic_distance2)
        self.assertEqual(
            disjoint_gene_value_connection + matching_genes_value_connection + disjoint_gene_value_node +
            matching_genes_value_node, genetic_distance)

    def test_calculate_genetic_distance_nodes(self):
        disjoint_gene_value = (3 / 6) * self.config.compatibility_factor_disjoint_genes
        matching_genes_value = 0.325 * self.config.compatibility_factor_matching_genes
        self.assertEqual(disjoint_gene_value + matching_genes_value,
                         ss._calculate_genetic_distance_nodes(self.g1, self.g2, self.config))

    def test_calculate_genetic_distance_connections(self):
        disjoint_gene_value = (3 / 7) * self.config.compatibility_factor_disjoint_genes
        matching_genes_value = 1.36 * self.config.compatibility_factor_matching_genes
        self.assertEqual(disjoint_gene_value + matching_genes_value,
                         ss._calculate_genetic_distance_connections(self.g1, self.g2, self.config))

    def test_sort_agents_into_species(self):
        species_id_generator = SpeciesIDGeneratorSingleCore()
        species_list_new = ss.sort_agents_into_species([], [self.agent1, self.agent2], species_id_generator,
                                                       self.config)
        self.assertEqual(2, len(species_list_new))

        # Test first species
        self.assertEqual(self.g1, species_list_new[0].representative)
        self.assertEqual(1, len(species_list_new[0].members))
        self.assertEqual(self.agent1, species_list_new[0].members[0])
        self.assertEqual(0, species_list_new[0].id_)

        # Test second species
        self.assertEqual(self.g2, species_list_new[1].representative)
        self.assertEqual(1, len(species_list_new[1].members))
        self.assertEqual(self.agent2, species_list_new[1].members[0])
        self.assertEqual(1, species_list_new[1].id_)

        # Insert one more matching genome
        genome_new = rs.deep_copy_genome(self.g1)
        agent_new = Agent(1, genome_new)

        species_list_new = ss.sort_agents_into_species(species_list_new, [agent_new], species_id_generator, self.config)
        self.assertEqual(2, len(species_list_new))
        # Test first species
        self.assertEqual(self.g1, species_list_new[0].representative)
        self.assertEqual(2, len(species_list_new[0].members))
        self.assertEqual(self.agent1, species_list_new[0].members[0])
        self.assertEqual(agent_new, species_list_new[0].members[1])
        self.assertEqual(0, species_list_new[0].id_)

        # Test second species
        self.assertEqual(self.g2, species_list_new[1].representative)
        self.assertEqual(1, len(species_list_new[1].members))
        self.assertEqual(self.agent2, species_list_new[1].members[0])
        self.assertEqual(1, species_list_new[1].id_)

    def test_update_fitness_species(self):
        generation = ss.update_fitness_species(self.generation)

        self.assertEqual(3, len(generation.species_list))
        self.assertEqual(3, generation.species_list[0].max_species_fitness)
        self.assertEqual(self.generation.number, generation.species_list[0].generation_max_species_fitness)

        self.assertEqual(7, generation.species_list[1].max_species_fitness)
        self.assertEqual(5, generation.species_list[1].generation_max_species_fitness)

        self.assertEqual(8, generation.species_list[2].max_species_fitness)
        self.assertEqual(self.generation.number, generation.species_list[2].generation_max_species_fitness)

    def test_get_allowed_species_for_reproduction(self):
        allowed_species = ss.get_allowed_species_for_reproduction(self.generation, 15)
        self.assertEqual(2, len(allowed_species))
        self.assertEqual(self.species1, allowed_species[0])
        self.assertEqual(self.species3, allowed_species[1])

    def test_calculate_adjusted_fitness(self):
        species = ss.calculate_adjusted_fitness(self.generation.species_list, min_fitness=1, max_fitness=8)
        self.assertEqual(3, len(species))
        self.assertEqual(1 / 7, species[0].adjusted_fitness)
        self.assertEqual(4 / 7, species[1].adjusted_fitness)
        self.assertEqual(6.5 / 7, species[2].adjusted_fitness)

        # Set all agents to negative fitness values
        self.agent1.fitness = -1
        self.agent2.fitness = -2
        self.agent3.fitness = -3
        self.agent4.fitness = -4
        self.agent5.fitness = -5
        self.agent6.fitness = -6
        self.agent7.fitness = -7
        self.agent8.fitness = -8

        species = ss.calculate_adjusted_fitness(self.generation.species_list, min_fitness=-8, max_fitness=-1)
        self.assertEqual(3, len(species))
        self.assertEqual(6 / 7, species[0].adjusted_fitness)
        self.assertEqual(3 / 7, species[1].adjusted_fitness)
        self.assertEqual(0.5 / 7, species[2].adjusted_fitness)

    def test_calculate_amount_offspring(self):
        self.species1.adjusted_fitness = 0.25
        self.species2.adjusted_fitness = 0.5
        self.species3.adjusted_fitness = 0.75

        offspring = ss.calculate_amount_offspring(self.generation.species_list, 19)
        self.assertEqual(3, len(offspring))
        self.assertEqual(4, offspring[0])
        self.assertEqual(6, offspring[1])
        self.assertEqual(9, offspring[2])
        self.assertEqual(19, sum(offspring))
