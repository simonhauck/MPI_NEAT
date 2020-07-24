from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import progressbar
from loguru import logger

from examples.BaseExample import BaseExample
from examples.pendulum.pendulum_challenge import PendulumChallenge
from neat_core.activation_function import sigmoid_activation
from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neural_network.basic_neural_network import BasicNeuralNetwork
from utils.fitness_evaluation import fitness_evaluation_utils
from utils.reporter.checkpoint_reporter import CheckPointReporter
from utils.reporter.fitness_reporter import FitnessReporter
from utils.reporter.species_reporter import SpeciesReporter
from utils.reporter.time_reporter import TimeReporter
from utils.visualization import text_visualization, reporter_visualization, genome_visualization


class PendulumOptimizer(BaseExample):

    def __init__(self) -> None:
        self.fitness_reporter = None
        self.species_reporter = None
        self.time_reporter = None
        self.check_point_reporter = None
        self.challenge = None

        self.progressbar = None
        self.progressbar_max = None
        self.progressbar_widgets = [
            'Generation: ', progressbar.Percentage(),
            ' ', progressbar.Bar(marker='#', left='[', right=']'),
            ' ', progressbar.Counter('Evaluated Agents: %(value)03d'),
            ', ', progressbar.ETA()
        ]

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
                            population_size=400,
                            compatibility_threshold=3,
                            weight_mutation_type="normal",
                            weight_mutation_normal_sigma=1.3,
                            connection_initial_min_weight=-5,
                            connection_initial_max_weight=5,
                            connection_min_weight=-15,
                            connection_max_weight=15,
                            bias_mutation_type="normal",
                            bias_mutation_normal_sigma=1.3,
                            bias_initial_min=-1,
                            bias_initial_max=1,
                            bias_min=-15,
                            bias_max=15,
                            compatibility_factor_disjoint_genes=1.0,
                            compatibility_factor_matching_genes=0.5,
                            probability_mutate_add_connection=0.5,
                            probability_mutate_add_node=0.2,
                            compatibility_genome_size_threshold=0)

        # Progressbar size
        self.progressbar_max = config.population_size

        # Create random seed, if none is specified
        if seed is None:
            seed = np.random.RandomState().randint(2 ** 24)
        logger.info("Used Seed: {}".format(seed))

        # Create the challenge
        self.challenge = PendulumChallenge()

        # Start evaluation
        optimizer.evaluate(amount_input_nodes=3,
                           amount_output_nodes=1,
                           activation_function=sigmoid_activation,
                           challenge=self.challenge,
                           config=config,
                           seed=seed)

    def on_initialization(self) -> None:
        logger.info("On initialization called...")

    def on_generation_evaluation_start(self, generation: Generation) -> None:
        logger.info(
            "Starting evaluation of generation {} with {} agents".format(generation.number, len(generation.agents)))
        self.progressbar = progressbar.ProgressBar(widgets=self.progressbar_widgets, max_value=self.progressbar_max)
        self.progressbar.start()

    def on_agent_evaluation_end(self, i: int, agent: Agent) -> None:
        self.progressbar.update(i + 1)

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        self.progressbar.finish()

        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        logger.info("Finished evaluation of generation {}".format(generation.number))
        logger.info("Best Fitness in Agent {}: {}, AdditionalInfo: {}".format(best_agent.id, best_agent.fitness,
                                                                              best_agent.additional_info))
        logger.info("Amount species: {}".format(len(generation.species_list)))

    def on_finish(self, generation: Generation) -> None:
        logger.info("OnFinish called with generation {}".format(generation.number))

        # Get the best agent
        agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        # Print genome
        text_visualization.print_agent(agent)

        # Plot fitness values
        reporter_visualization.plot_fitness_reporter(self.fitness_reporter.data, plot=True)

        # Plot species sizes
        reporter_visualization.plot_species_reporter(self.species_reporter.data, plot=True)

        # Plot required time values
        reporter_visualization.plot_time_reporter(self.time_reporter.data, plot=True)

        # Plot genome
        genome_visualization.draw_genome_graph(agent.genome, draw_labels=False)
        plt.show()

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
