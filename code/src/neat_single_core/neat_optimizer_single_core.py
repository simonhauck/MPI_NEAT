import numpy as np

from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.models.species import Species
from neat_core.optimizer.challenge import Challenge
from neat_core.optimizer.generator.inno_num_generator_interface import InnovationNumberGeneratorInterface
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer
from neat_core.service import generation_service as gs
from neat_core.service import reproduction_service as rp
from neat_core.service import species_service as ss
from neat_single_core.agent_id_generator_single_core import AgentIDGeneratorSingleCore
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore
from neat_single_core.species_id_generator_single_core import SpeciesIDGeneratorSingleCore
from neural_network.basic_neural_network import BasicNeuralNetwork


class NeatOptimizerSingleCore(NeatOptimizer):

    def evaluate(self, amount_input_nodes: int, amount_output_nodes,
                 activation_function, challenge: Challenge, config: NeatConfig,
                 seed: int) -> None:
        assert self.callback is not None

        # Initialize Parameters
        innovation_number_generator = InnovationNumberGeneratorSingleCore()
        species_id_generator = SpeciesIDGeneratorSingleCore()
        agent_id_generator = AgentIDGeneratorSingleCore()

        # Notify callback about starting evaluation
        self.callback.on_initialization()
        # Prepare challenge
        challenge.initialization()

        initial_generation = gs.create_initial_generation(amount_input_nodes, amount_output_nodes, activation_function,
                                                          innovation_number_generator, species_id_generator,
                                                          agent_id_generator, config, seed)

        finished_generation = self._evaluation_loop(initial_generation, challenge, innovation_number_generator,
                                                    species_id_generator, agent_id_generator, config)

        # Finish the evaluation and notify the callback
        self._cleanup(challenge)
        self.callback.on_finish(finished_generation)

    def evaluate_genome_structure(self, genome_structure: Genome, challenge: Challenge, config: NeatConfig, seed: int):
        assert self.callback is not None

        # Initialize Parameters
        innovation_number_generator = InnovationNumberGeneratorSingleCore()
        species_id_generator = SpeciesIDGeneratorSingleCore()
        agent_id_generator = AgentIDGeneratorSingleCore()

        # Notify callback about starting evaluation
        self.callback.on_initialization()
        # Prepare challenge
        challenge.initialization()

        initial_generation = gs.create_initial_generation_genome(genome_structure, innovation_number_generator,
                                                                 species_id_generator, agent_id_generator, config, seed)

        finished_generation = self._evaluation_loop(initial_generation, challenge, innovation_number_generator,
                                                    species_id_generator, agent_id_generator, config)

        # Finish the evaluation and notify the callback
        self._cleanup(challenge)
        self.callback.on_finish(finished_generation)

    def _evaluation_loop(self, generation: Generation, challenge: Challenge,
                         innovation_number_generator: InnovationNumberGeneratorInterface,
                         species_id_generator: SpeciesIDGeneratorSingleCore,
                         agent_id_generator: AgentIDGeneratorSingleCore,
                         config: NeatConfig) -> Generation:

        current_generation = generation
        while True:
            current_generation = self._evaluate_generation(current_generation, challenge)

            # Should e new generation be built?
            if self.callback.finish_evaluation(current_generation):
                break

            current_generation = self._build_new_generation(current_generation, innovation_number_generator,
                                                            species_id_generator, agent_id_generator, config)

        return current_generation

    def _evaluate_generation(self, generation: Generation, challenge: Challenge):
        # Notify callback
        self.callback.on_generation_evaluation_start(generation)

        for i, agent in zip(range(len(generation.agents)), generation.agents):
            # Prepare challenge and notify callback
            self.callback.on_agent_evaluation_start(i, agent)
            challenge.before_evaluation()

            # Create and build neural network
            neural_network = BasicNeuralNetwork()
            neural_network.build(agent.genome)

            # Evaluate agent, set values
            fitness, additional_info = challenge.evaluate(neural_network)
            agent.fitness = fitness
            agent.additional_info = additional_info

            # Postprocess challenge and notify callback
            challenge.after_evaluation()
            self.callback.on_agent_evaluation_end(i, agent)

        # Notify callback
        self.callback.on_generation_evaluation_end(generation)
        return generation

    def _build_new_generation(self, generation: Generation,
                              innovation_number_generator: InnovationNumberGeneratorInterface,
                              species_id_generator: SpeciesIDGeneratorSingleCore,
                              agent_id_generator: AgentIDGeneratorSingleCore,
                              config: NeatConfig) -> Generation:

        # Get the random generator for the new generation
        new_generation_seed = np.random.RandomState(generation.seed).randint(2 ** 24)
        rnd = np.random.RandomState(new_generation_seed)

        # Get the best agents, which will be copied later
        best_agents = gs.get_best_genomes_from_species(generation.species_list, 10)

        # Get allowed species for reproduction
        generation = ss.update_fitness_species(generation)
        # TODO config value!!!
        species_list = ss.get_allowed_species_for_reproduction(generation, 15)

        # Calculate the adjusted fitness values
        min_fitness = min([a.fitness for a in generation.agents])
        max_fitness = max([a.fitness for a in generation.agents])
        species_list = ss.calculate_adjusted_fitness(species_list, min_fitness, max_fitness)

        # TODO remove lower half
        # TODO reset innovation number id generator
        # Calculate off spring combinations

        # Calculate offspring for species
        off_spring_list = ss.calculate_amount_offspring(species_list, config.population_size - len(best_agents))

        new_agents = []
        rnd = np.random.RandomState()
        for agent in generation.agents:
            copied_genome = rp.deep_copy_genome(agent.genome)
            copied_genome = rp.mutate_weights(copied_genome, rnd, config)
            copied_genome = rp.mutate_bias(copied_genome, rnd, config)
            new_agents.append(Agent(1, copied_genome))

        return Generation(generation.number + 1, 0, new_agents, [Species(1, new_agents[0].genome, new_agents)])

    def _cleanup(self, challenge: Challenge) -> None:
        # Notify callback and challenge
        challenge.clean_up()
        self.callback.on_cleanup()
