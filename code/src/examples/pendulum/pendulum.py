import time
from datetime import datetime
from typing import List

import numpy as np
from loguru import logger

from examples.BaseExample import BaseExample
from examples.pendulum.pendulum_challenge import PendulumChallenge
from neat_core.activation_function import tanh_activation
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.optimizer.neat_reporter import NeatReporter
from neural_network.basic_neural_network import BasicNeuralNetwork
from utils.fitness_evaluation import fitness_evaluation_utils
from utils.reporter.checkpoint_reporter import CheckPointReporter
from utils.reporter.fitness_reporter import FitnessReporter
from utils.reporter.species_reporter import SpeciesReporter
from utils.reporter.time_reporter import TimeReporter
from utils.visualization import text_visualization


class PendulumOptimizer(BaseExample):

    def __init__(self) -> None:
        self.fitness_reporter = None
        self.species_reporter = None
        self.time_reporter = None
        self.check_point_reporter = None
        self.challenge = None

        self.start_time_generation = None

    def evaluate(self, optimizer: NeatOptimizer, seed: int = None, **kwargs):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Reporter for fitness values
        self.fitness_reporter = FitnessReporter()
        self.species_reporter = SpeciesReporter()
        self.time_reporter = TimeReporter()
        self.check_point_reporter = CheckPointReporter("/tmp/pendulum/{}/".format(time), "pendulum",
                                                       lambda _: True)

        # Register this class as callback
        optimizer.register_callback(self)

        optimizer.register_reporters(self.time_reporter,
                                     self.species_reporter,
                                     self.fitness_reporter,
                                     self.check_point_reporter)

        # Config
        config = NeatConfig(allow_recurrent_connections=False,
                            population_size=1000,
                            compatibility_threshold=4,
                            weight_mutation_type="normal",
                            weight_mutation_normal_sigma=0.5,
                            connection_initial_min_weight=-1,
                            connection_initial_max_weight=1,
                            connection_min_weight=-15,
                            connection_max_weight=15,
                            bias_mutation_type="normal",
                            bias_mutation_normal_sigma=0.5,
                            bias_initial_min=-1,
                            bias_initial_max=1,
                            bias_min=-15,
                            bias_max=15,
                            compatibility_factor_disjoint_genes=1.0,
                            compatibility_factor_matching_genes=4.0,
                            probability_mutate_add_connection=0.5,
                            probability_mutate_add_node=0.2,
                            compatibility_genome_size_threshold=0)

        # Create random seed, if none is specified
        if seed is None:
            seed = np.random.RandomState().randint(2 ** 24)
        logger.info("Used Seed: {}".format(seed))

        # Create the challenge
        self.challenge = PendulumChallenge()

        # Start evaluation
        optimizer.evaluate(amount_input_nodes=3,
                           amount_output_nodes=1,
                           activation_function=tanh_activation,
                           challenge=self.challenge,
                           config=config,
                           seed=seed)

    def on_initialization(self) -> None:
        logger.info("On initialization called...")

    def on_generation_evaluation_start(self, generation: Generation) -> None:
        logger.info(
            "Starting evaluation of generation {} with {} agents".format(generation.number, len(generation.agents)))
        self.start_time_generation = time.time()

    def on_generation_evaluation_end(self, generation: Generation, reporters: List[NeatReporter]) -> None:
        required_time = round(time.time() - self.start_time_generation, 4)
        logger.info("Finished evaluation of generation {}, Required Time: {}".format(generation.number, required_time))

        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        logger.info("Best Fitness in Agent {}: {}, AdditionalInfo: {}".format(best_agent.id, best_agent.fitness,
                                                                              best_agent.additional_info))
        logger.info("Amount species: {}".format(len(generation.species_list)))

    def on_finish(self, generation: Generation, reporters: List[NeatReporter]) -> None:
        logger.info("OnFinish called with generation {}".format(generation.number))

        # Get the best agent
        agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        logger.info("Best Agent - Fitness:  {}, AdditionalInfo: {}".format(agent.fitness, agent.additional_info))

        # Print genome
        text_visualization.print_agent(agent)

    def finish_evaluation(self, generation: Generation) -> bool:
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        return best_agent.additional_info["solved"]

    def visualize_genome(self, genome: Genome, **kwargs) -> None:
        nn = BasicNeuralNetwork()
        nn.build(genome)

        challenge = PendulumChallenge()
        challenge.initialization(show=True)

        challenge.before_evaluation(show=True)
        fitness, additional_info = challenge.evaluate(nn, show=True)
        challenge.clean_up(show=True)
        logger.info("Finished Neural network  Fitness: {}, Info: {}".format(fitness, additional_info))
