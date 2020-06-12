import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from examples.xor.xor_challenge import ChallengeXOR
from neat_core.activation_function import modified_sigmoid_activation
from neat_core.models.agent import Agent
from neat_core.models.connection import Connection
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.optimizer.neat_optimizer_callback import NeatOptimizerCallback
from neural_network.basic_neural_network import BasicNeuralNetwork
from utils.fitness_evaluation import fitness_evaluation_utils
from utils.reporter import fitness_reporter, species_reporter
from utils.visualization import genome_visualization
from utils.visualization import reporter_visualization
from utils.visualization import text_visualization


class XOROptimizer(NeatOptimizerCallback):

    def __init__(self) -> None:
        self.fitness_reporter = None
        self.solved_generation_number = None
        self.species_reporter = None

    def evaluate(self, optimizer: NeatOptimizer):
        self.fitness_reporter = fitness_reporter.FitnessReporter()
        self.species_reporter = species_reporter.SpeciesReporter()

        # Register this class as callback
        optimizer.register_callback(self)
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

        # config = NeatConfig(allow_recurrent_connections=False,
        #                     population_size=150,
        #                     compatibility_threshold=3,
        #                     connection_min_weight=-15,
        #                     connection_max_weight=15,
        #                     bias_min=-15,
        #                     bias_max=15,
        #                     compatibility_factor_disjoint_genes=1.0,
        #                     compatibility_factor_matching_genes=0.4,
        #                     probability_mutate_add_connection=0.05,
        #                     probability_mutate_add_node=0.03)

        # Specify the config
        config = NeatConfig(allow_recurrent_connections=False,
                            population_size=150,
                            compatibility_threshold=3,
                            weight_mutation_type="normal",
                            weight_mutation_normal_sigma=1.3,
                            connection_initial_min_weight=-5,
                            connection_initial_max_weight=5,
                            connection_min_weight=-5,
                            connection_max_weight=5,
                            bias_mutation_type="normal",
                            bias_mutation_normal_sigma=1.3,
                            bias_initial_min=-1,
                            bias_initial_max=1,
                            bias_min=-5,
                            bias_max=5,
                            compatibility_factor_disjoint_genes=1.0,
                            compatibility_factor_matching_genes=0.5,
                            probability_mutate_add_connection=0.5,
                            probability_mutate_add_node=0.2,
                            compatibility_genome_size_threshold=0)

        seed = np.random.RandomState().randint(2 ** 24)
        logger.info("Used Seed: {}".format(seed))

        optimizer.evaluate(amount_input_nodes=2,
                           amount_output_nodes=1,
                           activation_function=modified_sigmoid_activation,
                           challenge=ChallengeXOR(),
                           config=config,
                           seed=seed)

    # TODO remove at the end
    def evaluate_fix_structure(self, optimizer: NeatOptimizer):
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

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        fitness_reporter.add_generation_fitness_reporter(self.fitness_reporter, generation)
        species_reporter.add_generation_species_reporter(self.species_reporter, generation)

        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        logger.info("Finished evaluation of generation {}".format(generation.number))
        logger.info("Best Fitness in Agent {}: {}".format(best_agent.id, best_agent.fitness))
        logger.info("Amount species: {}".format(len(generation.species_list)))

    def on_cleanup(self) -> None:
        logger.info("Cleanup called...")

    def on_finish(self, generation: Generation) -> None:
        logger.info("OnFinish called with generation {}".format(generation.number))
        self.solved_generation_number = generation.number

        # Get the best agent
        agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        # Print genome
        text_visualization.print_agent(agent)

        # Plot fitness curve
        reporter_visualization.plot_fitness_reporter(self.fitness_reporter, plot=True)

        # Plot species sizes
        reporter_visualization.plot_species_reporter(self.species_reporter, plot=True)

        # Draw genome
        genome_visualization.draw_genome_graph(agent.genome, draw_labels=False)
        plt.show()

        # Print actual results
        challenge = ChallengeXOR()
        nn = BasicNeuralNetwork()
        nn.build(agent.genome)
        fitness, additional_info = challenge.evaluate(nn, show=True)
        logger.info(
            "Finished Evaluation - Fitness: {}, Challenge solved = {}".format(fitness, additional_info["solved"]))

    def finish_evaluation(self, generation: Generation) -> bool:
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        return best_agent.additional_info["solved"] or generation.number >= 500
