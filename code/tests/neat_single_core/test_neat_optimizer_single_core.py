from typing import Dict
from unittest import TestCase

from neat_core.activation_function import step_activation, modified_sigmoid_activation
from neat_core.models.agent import Agent
from neat_core.models.connection import Connection
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.challenge import Challenge
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer_callback import NeatOptimizerCallback
from neat_core.service import generation_service as gs
from neat_single_core.agent_id_generator_single_core import AgentIDGeneratorSingleCore
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore
from neat_single_core.neat_optimizer_single_core import NeatOptimizerSingleCore
from neat_single_core.species_id_generator_single_core import SpeciesIDGeneratorSingleCore
from neural_network.neural_network_interface import NeuralNetworkInterface


class MockCallback(NeatOptimizerCallback):

    def __init__(self) -> None:
        self.on_initialization_count = 0
        self.on_reproduction_start_count = 0
        self.on_reproduction_end_count = 0
        self.on_compose_offsprings_start_count = 0
        self.on_compose_offsprings_end_count = 0
        self.on_generation_evaluation_start_count = 0
        self.on_agent_evaluation_start_count = 0
        self.on_agent_evaluation_end_count = 0
        self.on_generation_evaluation_end_count = 0
        self.on_cleanup_count = 0
        self.on_finish_count = 0
        self.finish_evaluation_count = 0
        self.finish_generation: Generation = None

    def on_initialization(self) -> None:
        self.on_initialization_count += 1

    def on_reproduction_start(self, generation: Generation) -> None:
        self.on_reproduction_start_count += 1

    def on_compose_offsprings_start(self) -> None:
        self.on_compose_offsprings_start_count += 1

    def on_compose_offsprings_end(self) -> None:
        self.on_compose_offsprings_end_count += 1

    def on_reproduction_end(self, generation: Generation) -> None:
        self.on_reproduction_end_count += 1

    def on_generation_evaluation_start(self, generation: Generation) -> None:
        self.on_generation_evaluation_start_count += 1

    def on_agent_evaluation_start(self, i: int, agent: Agent) -> None:
        self.on_agent_evaluation_start_count += 1

    def on_agent_evaluation_end(self, i: int, agent: Agent) -> None:
        self.on_agent_evaluation_end_count += 1

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        self.on_generation_evaluation_end_count += 1

    def on_cleanup(self) -> None:
        self.on_cleanup_count += 1

    def on_finish(self, generation: Generation) -> None:
        self.on_finish_count += 1
        self.finish_generation = generation

    def finish_evaluation(self, generation: Generation) -> bool:
        if self.finish_evaluation_count >= 10:
            return True
        else:
            self.finish_evaluation_count += 1


class MockChallenge(Challenge):

    def __init__(self) -> None:
        self.initialization_count = 0
        self.before_evaluation_count = 0
        self.evaluate_count = 0
        self.after_evaluation_count = 0
        self.clean_up_count = 0

    def initialization(self) -> None:
        self.initialization_count += 1

    def before_evaluation(self) -> None:
        self.before_evaluation_count += 1

    def evaluate(self, neural_network: NeuralNetworkInterface, **kwargs) -> (float, Dict[str, object]):
        fitness = self.evaluate_count
        self.evaluate_count += 1
        return fitness, {}

    def after_evaluation(self) -> None:
        self.after_evaluation_count += 1

    def clean_up(self):
        self.clean_up_count += 1


class NeatOptimizerSingleCoreTest(TestCase):

    def setUp(self) -> None:
        self.optimizer_single = NeatOptimizerSingleCore()
        self.callback = MockCallback()
        self.challenge = MockChallenge()

        self.optimizer_single.register_callback(self.callback)
        self.config = NeatConfig(population_size=150)

    def test_subscribe(self):
        self.optimizer_single.unregister_callback()

        self.assertIsNone(self.optimizer_single.callback)
        self.optimizer_single.register_callback(self.callback)
        self.assertEqual(self.callback, self.optimizer_single.callback)

    def test_unsubscribe(self):
        self.assertEqual(self.callback, self.optimizer_single.callback)
        self.optimizer_single.unregister_callback()
        self.assertIsNone(self.optimizer_single.callback)

    def test_evaluate(self):
        self.optimizer_single.evaluate(3, 2, step_activation, self.challenge, self.config, 1)

        expected_generation_number = 10

        # Test callback functions
        self.assertEqual(1, self.challenge.initialization_count)

        # Reproduction functions
        self.assertEqual(expected_generation_number, self.callback.on_reproduction_start_count)
        self.assertEqual(expected_generation_number, self.callback.on_compose_offsprings_start_count)
        self.assertEqual(expected_generation_number, self.callback.on_compose_offsprings_end_count)
        self.assertEqual(expected_generation_number, self.callback.on_reproduction_start_count)

        # Evaluation functions
        self.assertEqual(expected_generation_number + 1, self.callback.on_generation_evaluation_start_count)
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.callback.on_agent_evaluation_start_count)
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.callback.on_agent_evaluation_end_count)
        self.assertEqual(expected_generation_number + 1, self.callback.on_generation_evaluation_end_count)
        self.assertEqual(expected_generation_number, self.callback.finish_evaluation_count)

        self.assertEqual(1, self.callback.on_finish_count)
        self.assertEqual(1, self.callback.on_cleanup_count)

        # Check generated networks
        self.assertEqual(self.config.population_size, len(self.callback.finish_generation.agents))

    def test_evaluate_genome_structure(self):
        genome = Genome(
            0,
            [Node(0, NodeType.INPUT, 0, modified_sigmoid_activation, 0),
             Node(1, NodeType.INPUT, 0, modified_sigmoid_activation, 0),
             Node(2, NodeType.OUTPUT, 0, modified_sigmoid_activation, 1),
             Node(3, NodeType.HIDDEN, 0, modified_sigmoid_activation, 0.5),
             Node(4, NodeType.HIDDEN, 0, modified_sigmoid_activation, 0.5)],
            [Connection(1, 0, 3, 0.1, True),
             Connection(2, 1, 3, 0.1, True),
             Connection(3, 0, 4, 0.1, True),
             Connection(4, 1, 4, 0.1, True),
             Connection(5, 3, 2, 0.1, True),
             Connection(6, 4, 2, 0.1, True)]
        )
        self.optimizer_single.evaluate_genome_structure(genome, self.challenge, self.config, 1)

        expected_generation_number = 10

        # Test callback functions
        self.assertEqual(1, self.challenge.initialization_count)

        # Reproduction functions
        self.assertEqual(expected_generation_number, self.callback.on_reproduction_start_count)
        self.assertEqual(expected_generation_number, self.callback.on_compose_offsprings_start_count)
        self.assertEqual(expected_generation_number, self.callback.on_compose_offsprings_end_count)
        self.assertEqual(expected_generation_number, self.callback.on_reproduction_start_count)

        # Evaluation loop
        self.assertEqual(expected_generation_number + 1, self.callback.on_generation_evaluation_start_count)
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.callback.on_agent_evaluation_start_count)
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.callback.on_agent_evaluation_end_count)
        self.assertEqual(expected_generation_number + 1, self.callback.on_generation_evaluation_end_count)
        self.assertEqual(expected_generation_number, self.callback.finish_evaluation_count)

        # Finish evaluation
        self.assertEqual(1, self.callback.on_finish_count)
        self.assertEqual(1, self.callback.on_cleanup_count)

        # Check generated networks
        self.assertEqual(self.config.population_size, len(self.callback.finish_generation.agents))

    def test_evaluation_loop(self):
        inno_generator = InnovationNumberGeneratorSingleCore()
        species_id_generator = SpeciesIDGeneratorSingleCore()
        agent_id_generator = AgentIDGeneratorSingleCore()

        initial_generation = gs.create_initial_generation(2, 1, step_activation, inno_generator, species_id_generator,
                                                          agent_id_generator, self.config, 1)
        final_generation = self.optimizer_single._evaluation_loop(initial_generation, self.challenge, inno_generator,
                                                                  species_id_generator, agent_id_generator, self.config)

        expected_generation_number = 10
        self.assertEqual(expected_generation_number, final_generation.number)
        self.assertEqual(self.config.population_size, len(final_generation.agents))

        # Check genome fitness
        for i, agent in zip(range(len(final_generation.agents)), final_generation.agents):
            self.assertEqual(expected_generation_number * self.config.population_size + i, agent.fitness)

        # Check challenge calls
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.challenge.before_evaluation_count)
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.challenge.evaluate_count)
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.challenge.after_evaluation_count)

        # Check callback calls
        self.assertEqual(expected_generation_number + 1, self.callback.on_generation_evaluation_start_count)
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.callback.on_agent_evaluation_start_count)
        self.assertEqual((1 + expected_generation_number) * self.config.population_size,
                         self.callback.on_agent_evaluation_end_count)
        self.assertEqual(expected_generation_number + 1, self.callback.on_generation_evaluation_end_count)

    def test_evaluate_generation(self):
        generation = gs.create_initial_generation(2, 1, step_activation, InnovationNumberGeneratorSingleCore(),
                                                  SpeciesIDGeneratorSingleCore(), AgentIDGeneratorSingleCore(),
                                                  self.config, 1)

        self.optimizer_single._evaluate_generation(generation, self.challenge)

        # Check challenge
        self.assertEqual(self.config.population_size, self.challenge.before_evaluation_count)
        self.assertEqual(self.config.population_size, self.challenge.evaluate_count)
        self.assertEqual(self.config.population_size, self.challenge.after_evaluation_count)

        # Check callback
        self.assertEqual(1, self.callback.on_generation_evaluation_start_count)
        self.assertEqual(self.config.population_size, self.callback.on_agent_evaluation_start_count)
        self.assertEqual(self.config.population_size, self.callback.on_agent_evaluation_end_count)
        self.assertEqual(1, self.callback.on_generation_evaluation_end_count)

        for expected_fitness, agent in zip(range(self.config.population_size), generation.agents):
            self.assertEqual(expected_fitness, agent.fitness)

    def test_cleanup(self):
        self.optimizer_single._cleanup(self.challenge)
        self.assertEqual(1, self.challenge.clean_up_count)
        self.assertEqual(1, self.callback.on_cleanup_count)
