from datetime import datetime
from typing import List

import numpy as np
from loguru import logger

from examples.BaseExample import BaseExample
from examples.xor.xor_challenge import ChallengeXOR
from neat_core.activation_function import modified_sigmoid_activation
from neat_core.models.agent import Agent
from neat_core.models.connection import Connection
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
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


class XOROptimizer(BaseExample):

    def __init__(self) -> None:
        self.fitness_reporter: FitnessReporter = None
        self.species_reporter: SpeciesReporter = None
        self.time_reporter: TimeReporter = None
        self.check_point_reporter: CheckPointReporter = None
        self.solved_generation_number = None

    def evaluate(self, optimizer: NeatOptimizer, seed: int = None, **kwargs):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Create reporter
        self.fitness_reporter = FitnessReporter()
        self.species_reporter = SpeciesReporter()
        self.time_reporter = TimeReporter()
        self.check_point_reporter = CheckPointReporter("/tmp/xor/{}/".format(time), "xor_genome_",
                                                       lambda _: False)

        # Register this class as callback and reporter
        optimizer.register_callback(self)
        optimizer.register_reporters(self.time_reporter,
                                     self.fitness_reporter,
                                     self.species_reporter,
                                     self.check_point_reporter)

        # Normal config
        # config = NeatConfig(allow_recurrent_connections=False,
        #                     population_size=150,
        #                     compatibility_threshold=3,
        #                     connection_min_weight=-15,
        #                     connection_max_weight=15,
        #                     bias_min=-15,
        #                     bias_max=15,
        #                     compatibility_factor_disjoint_genes=1.0,
        #                     compatibility_factor_matching_genes=0.5,
        #                     probability_mutate_add_connection=0.5,
        #                     probability_mutate_add_node=0.2)

        config = NeatConfig(allow_recurrent_connections=False,
                            population_size=150,
                            compatibility_threshold=3,
                            connection_initial_min_weight=-2,
                            connection_initial_max_weight=2,
                            bias_initial_min=-2,
                            bias_initial_max=2,
                            connection_min_weight=-6,
                            connection_max_weight=6,
                            weight_mutation_normal_sigma=3,
                            bias_min=-6,
                            bias_max=6,
                            compatibility_genome_size_threshold=16,
                            compatibility_factor_disjoint_genes=1.0,
                            compatibility_factor_matching_genes=0.4,
                            probability_mutate_add_connection=0.05,
                            probability_mutate_add_node=0.03)

        # Standard paper config
        # config = NeatConfig(allow_recurrent_connections=False,
        #                     population_size=150,
        #                     compatibility_threshold=3,
        #                     connection_min_weight=-15,
        #                     connection_max_weight=15,
        #                     connection_initial_min_weight=-1,
        #                     connection_initial_max_weight=1,
        #                     weight_mutation_normal_sigma=1.5,
        #                     bias_min=0,
        #                     bias_max=0,
        #                     bias_initial_min=0,
        #                     bias_initial_max=0,
        #                     compatibility_factor_disjoint_genes=1.0,
        #                     compatibility_factor_matching_genes=0.4,
        #                     probability_mutate_add_connection=0.5,
        #                     probability_mutate_add_node=0.2,
        #                     compatibility_genome_size_threshold=0)

        # Create random seed, if none is specified
        if seed is None:
            seed = np.random.RandomState().randint(2 ** 24)
        logger.info("Used Seed: {}".format(seed))

        optimizer.evaluate(amount_input_nodes=2,
                           amount_output_nodes=1,
                           activation_function=modified_sigmoid_activation,
                           challenge=ChallengeXOR(),
                           config=config,
                           seed=seed)

    # TODO remove at the end
    def evaluate_fix_structure(self, optimizer: NeatOptimizer, seed: int = None):
        optimizer.register_callback(self)
        config = NeatConfig(allow_recurrent_connections=False,
                            population_size=150,
                            compatibility_threshold=3,
                            connection_min_weight=-5,
                            connection_max_weight=5,
                            bias_min=-5,
                            bias_max=5,
                            probability_mutate_add_node=0,
                            probability_mutate_add_connection=0)

        # Create random seed, if none is specified
        if seed is None:
            seed = np.random.RandomState().randint(2 ** 24)
        logger.info("Used Seed: {}".format(seed))

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

        optimizer.evaluate_genome_structure(genome_structure=genome,
                                            challenge=ChallengeXOR(),
                                            config=config,
                                            seed=seed)

    def on_initialization(self) -> None:
        logger.info("On initialization called...")

    def on_generation_evaluation_start(self, generation: Generation) -> None:
        logger.info(
            "Starting evaluation of generation {} with {} agents".format(generation.number, len(generation.agents)))

    def on_agent_evaluation_start(self, i: int, agent: Agent) -> None:
        logger.debug("Starting evaluation of agent {}...".format(i))

    def on_agent_evaluation_end(self, i: int, agent: Agent) -> None:
        logger.debug("Finished evaluation of agent {} with fitness {}".format(i, agent.fitness))

    def on_generation_evaluation_end(self, generation: Generation, reporters: List[NeatReporter]) -> None:
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        logger.info("Finished evaluation of generation {}".format(generation.number))
        logger.info("Best Fitness in Agent {}: {}".format(best_agent.id, best_agent.fitness))
        logger.info("Amount species: {}".format(len(generation.species_list)))

    def on_cleanup(self) -> None:
        logger.info("Cleanup called...")

    def on_finish(self, generation: Generation, reporters: List[NeatReporter]) -> None:
        logger.info("OnFinish called with generation {}".format(generation.number))
        self.solved_generation_number = generation.number

        # Get the best agent
        agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        # Print genome
        text_visualization.print_agent(agent)

        # # Plot fitness curve
        # reporter_visualization.plot_fitness_reporter(self.fitness_reporter.data, plot=True)
        #
        # # Plot species sizes
        # reporter_visualization.plot_species_reporter(self.species_reporter.data, plot=True)
        #
        # # Plot required time
        # reporter_visualization.plot_time_reporter(self.time_reporter.data, plot=True)
        #
        # # Draw genome
        # genome_visualization.draw_genome_graph(agent.genome, draw_labels=False)
        # plt.show()

        # Visualize results
        self.visualize_genome(agent.genome)

    def finish_evaluation(self, generation: Generation) -> bool:
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        return best_agent.additional_info["solved"] or generation.number >= 500

    def visualize_genome(self, genome: Genome, **kwargs) -> None:
        nn = BasicNeuralNetwork()
        nn.build(genome)

        challenge = ChallengeXOR()

        fitness, additional_info = challenge.evaluate(nn, show=True)

        logger.info(
            "Finished Evaluation - Fitness: {}, Challenge solved = {}".format(fitness, additional_info["solved"]))
