import matplotlib.pyplot as plt
import numpy as np
import progressbar
from loguru import logger

from examples.mountain_car.mountain_car_challenge import ChallengeMountainCar
from neat_core.activation_function import modified_sigmoid_activation
from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.optimizer.neat_optimizer_callback import NeatOptimizerCallback
from neural_network.basic_neural_network import BasicNeuralNetwork
from utils.fitness_evaluation import fitness_evaluation_utils
from utils.reporter import fitness_reporter, species_reporter
from utils.visualization import genome_visualization
from utils.visualization import reporter_visualization
from utils.visualization import text_visualization


class MountainCarOptimizer(NeatOptimizerCallback):

    def __init__(self) -> None:
        self.fitness_reporter = None
        self.species_reporter = None
        self.challenge = None

        self.progressbar = None
        self.progressbar_max = None
        self.progressbar_widgets = [
            'Generation: ', progressbar.Percentage(),
            ' ', progressbar.Bar(marker='#', left='[', right=']'),
            ' ', progressbar.Counter('Evaluated Agents: %(value)03d'),
            ', ', progressbar.ETA()
        ]

    def evaluate(self, optimizer: NeatOptimizer):
        # Reporter for fitness values
        self.fitness_reporter = fitness_reporter.FitnessReporter()
        self.species_reporter = species_reporter.SpeciesReporter()

        # Register this class as callback
        optimizer.register_callback(self)

        # Config
        config = NeatConfig(allow_recurrent_connections=True,
                            population_size=300,
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

        # Progressbar size
        self.progressbar_max = config.population_size

        # Set seed
        seed = np.random.RandomState().randint(2 ** 24)
        logger.info("Used Seed: {}".format(seed))

        # Create the challenge
        self.challenge = ChallengeMountainCar()

        # Start evaluation
        optimizer.evaluate(amount_input_nodes=2,
                           amount_output_nodes=3,
                           activation_function=modified_sigmoid_activation,
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

    def on_agent_evaluation_start(self, i: int, agent: Agent) -> None:
        logger.debug("Starting evaluation of agent {}...".format(i))

    def on_agent_evaluation_end(self, i: int, agent: Agent) -> None:
        logger.debug("Finished evaluation of agent {} with fitness {}".format(i, agent.fitness))
        self.progressbar.update(i + 1)

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        self.progressbar.finish()

        # Add generation to fitness reporter
        fitness_reporter.add_generation_fitness_reporter(self.fitness_reporter, generation)
        species_reporter.add_generation_species_reporter(self.species_reporter, generation)

        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        logger.info("Finished evaluation of generation {}".format(generation.number))
        logger.info("Best Fitness in Agent {}: {}, AdditionalInfo: {}".format(best_agent.id, best_agent.fitness,
                                                                              best_agent.additional_info))
        logger.info("Amount species: {}".format(len(generation.species_list)))

    def on_cleanup(self) -> None:
        logger.info("Cleanup called...")

    def on_finish(self, generation: Generation) -> None:
        logger.info("OnFinish called with generation {}".format(generation.number))

        # Get the best agent
        agent = fitness_evaluation_utils.get_best_agent(generation.agents)

        # Print genome
        text_visualization.print_agent(agent)

        # Plot fitness values
        reporter_visualization.plot_fitness_reporter(self.fitness_reporter, plot=True)

        # Plot species sizes
        reporter_visualization.plot_species_reporter(self.species_reporter, plot=True)

        # Plot genome
        genome_visualization.draw_genome_graph(agent.genome, draw_labels=False)
        plt.show()

        # Run best genome in endless loop
        nn = BasicNeuralNetwork()
        nn.build(agent.genome)

        challenge = ChallengeMountainCar()
        challenge.initialization(show=True)

        # for i in range(100):
        while True:
            nn.reset()
            challenge.before_evaluation(show=True)
            fitness, additional_info = challenge.evaluate(nn, show=True)
            logger.info("Finished Neural network  Fitness: {}, Info: {}".format(fitness, additional_info))

        challenge.clean_up(show=True)

    def finish_evaluation(self, generation: Generation) -> bool:
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        return best_agent.fitness >= 90 ** 2
        # return best_agent.additional_info["solved"]
        # return generation.number >= 40
