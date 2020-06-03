import matplotlib.pyplot as plt
import numpy as np
import progressbar
from loguru import logger

from examples.mountain_car.mountain_car_challenge import ChallengeMountainCar
from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.optimizer.neat_optimizer_callback import NeatOptimizerCallback
from neural_network.basic_neural_network import BasicNeuralNetwork
from utils.fitness_evaluation import fitness_evaluation_utils
from utils.visualization import generation_visualization
from utils.visualization import genome_visualization
from utils.visualization import text_visualization


class MountainCarOptimizer(NeatOptimizerCallback):

    def __init__(self) -> None:
        self.plot_data = generation_visualization.PlotData()
        self.challenge = None

        self.progressbar = None
        self.progressbar_widgets = [
            'Generation: ', progressbar.Percentage(),
            ' ', progressbar.Bar(marker='#', left='[', right=']'),
            ' ', progressbar.Counter('Evaluated Agents: %(value)03d'),
            ', ', progressbar.ETA()
        ]
        self.progressbar_max = 0

    def evaluate(self, optimizer: NeatOptimizer):
        # Good seed: 11357659, pop 400, threshold 1.0, min max weight = -3,3

        optimizer.register_callback(self)
        config = NeatConfig(population_size=400, compatibility_threshold=1.0,
                            connection_min_weight=-3, connection_max_weight=3)

        self.progressbar_max = config.population_size

        seed = np.random.RandomState().randint(2 ** 24)
        logger.info("Used Seed: {}".format(seed))

        self.challenge = ChallengeMountainCar()

        optimizer.evaluate(amount_input_nodes=2, amount_output_nodes=3, activation_function=modified_sigmoid_function,
                           challenge=self.challenge, config=config, seed=seed)

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
        best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        self.plot_data.add_generation(generation)

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

        # Draw genome
        generation_visualization.plot_fitness_values(self.plot_data)
        plt.show()
        genome_visualization.draw_genome_graph(agent.genome, draw_labels=False)
        plt.show()

        # Run best genome in endless loop
        nn = BasicNeuralNetwork()
        nn.build(agent.genome)

        challenge = ChallengeMountainCar()
        challenge.initialization()

        while True:
            challenge.before_evaluation()
            fitness, additional_info = challenge.evaluate(nn, show=False)
            logger.info("Finished Neural network  Fitness: {}, Info: {}".format(fitness, additional_info))

    def finish_evaluation(self, generation: Generation) -> bool:
        # best_agent = fitness_evaluation_utils.get_best_agent(generation.agents)
        # return best_agent.additional_info["solved"]
        return generation.number >= 40
