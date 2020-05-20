import numpy as np

from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.agent import Agent
from neat_core.models.connection import Connection
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.models.node import Node, NodeType
from neat_core.optimizer.challenge import Challenge
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.service import generation_service as gs
from neat_core.service import reproduction_service as rp
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore
from neural_network.basic_neural_network import BasicNeuralNetwork


class NeatOptimizerSingleCore(NeatOptimizer):

    def __init__(self):
        super().__init__()
        self.current_generation: Generation = None
        self.innovation_number_generator: InnovationNumberGeneratorInterface = None
        self.config: NeatConfig = None
        self.challenge: Challenge = None

    def start_evaluation(self, amount_input_nodes: int, amount_output_nodes,
                         activation_function, challenge: Challenge, config: NeatConfig,
                         seed: int = 1) -> None:
        self.innovation_number_generator = InnovationNumberGeneratorSingleCore()
        self.config = config
        self.challenge = challenge

        # Notify callback about starting evaluation
        self.callback.on_initialization()
        # Prepare challenge
        self.challenge.initialization()

        # self.current_generation = gs.create_initial_generation(amount_input_nodes, amount_output_nodes,
        #                                                        activation_function,
        #                                                        self.innovation_number_generator,
        #                                                        self.config,
        #                                                        seed)

        genome = Genome(
            0, 0,
            [Node(0, NodeType.INPUT, 0, modified_sigmoid_function, 0),
             Node(1, NodeType.INPUT, 0, modified_sigmoid_function, 0),
             Node(2, NodeType.OUTPUT, 0, modified_sigmoid_function, 1),
             Node(3, NodeType.HIDDEN, 0, modified_sigmoid_function, 0.5),
             Node(4, NodeType.HIDDEN, 0, modified_sigmoid_function, 0.5)],
            [Connection(1, 0, 3, 0, True),
             Connection(2, 1, 3, 0, True),
             Connection(3, 0, 4, 0, True),
             Connection(4, 1, 4, 0, True),
             Connection(5, 3, 2, 0, True),
             Connection(6, 4, 2, 0, True)]
        )
        self.current_generation = gs.create_initial_generation_genome(genome, self.innovation_number_generator,
                                                                      self.config, seed)

        self._evaluate_generation()

    def evaluate_next_generation(self):
        next_generation_number = self.current_generation.number + 1

        new_agents = []
        rnd = np.random.RandomState()
        for agent in self.current_generation.agents:
            copied_genome = rp.deep_copy_genome(agent.genome)
            copied_genome = rp.mutate_weights(copied_genome, rnd, self.config)
            copied_genome = rp.mutate_bias(copied_genome, rnd, self.config)
            new_agents.append(Agent(copied_genome))

        self.current_generation = Generation(next_generation_number, 0, new_agents, [])

        self._evaluate_generation()

    def cleanup(self) -> None:
        # Notify callback and challenge
        self.challenge.clean_up()
        self.callback.on_cleanup()

    def _evaluate_generation(self):
        # Notify callback
        self.callback.on_generation_evaluation_start(self.current_generation)

        for agent in self.current_generation.agents:
            # Prepare challenge and notify callback
            self.callback.on_agent_evaluation_start(agent)
            self.challenge.before_evaluation()

            # Create and build neural network
            neural_network = BasicNeuralNetwork()
            neural_network.build(agent.genome)

            # Evaluate agent, set values
            fitness, additional_info = self.challenge.evaluate(neural_network)
            agent.fitness = fitness
            agent.additional_info = additional_info

            # Postprocess challenge and notify callback
            self.challenge.after_evaluation()
            self.callback.on_agent_evaluation_end(agent)

        # Notify callback
        self.callback.on_generation_evaluation_end(self.current_generation)
