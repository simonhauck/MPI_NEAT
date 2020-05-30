import numpy as np

from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
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
        best_agents_genomes = gs.get_best_genomes_from_species(generation.species_list, 10)

        # Get allowed species for reproduction
        generation = ss.update_fitness_species(generation)
        # TODO config value!!!
        species_list = ss.get_allowed_species_for_reproduction(generation, 15)

        # TODO handle error no species
        if len(species_list) <= 3:
            species_list = generation.species_list
        # assert len(species_list) >= 1

        # Calculate the adjusted fitness values
        min_fitness = min([a.fitness for a in generation.agents])
        max_fitness = max([a.fitness for a in generation.agents])
        species_list = ss.calculate_adjusted_fitness(species_list, min_fitness, max_fitness)

        # TODO config value!
        # Remove the low performing genomes
        species_list = ss.remove_low_genomes(species_list, 0.5)

        # Calculate offspring for species
        off_spring_list = ss.calculate_amount_offspring(species_list, config.population_size - len(best_agents_genomes))

        # Calculate off spring combinations
        off_spring_pairs = []
        for species, amount_offspring in zip(species_list, off_spring_list):
            off_spring_pairs += ss.create_offspring_pairs(species, amount_offspring, agent_id_generator, generation,
                                                          rnd, config)

        # Notify innovation number generator, that a new generation is created
        innovation_number_generator.next_generation(generation.number)

        # Create a dictionary for easy access
        agent_dict = {agent.id: agent for agent in generation.agents}

        # Create new agents - fill initially with best agents
        new_agents = [Agent(agent_id_generator.get_agent_id(), genome) for genome in best_agents_genomes]

        # Create agents with crossover
        for parent1_id, parent2_id, child_id in off_spring_pairs:
            parent1 = agent_dict[parent1_id]
            parent2 = agent_dict[parent2_id]

            child_seed = (parent1.genome.seed + parent2.genome.seed) % 2 ** 24
            rnd_child = np.random.RandomState(child_seed)

            # Perform crossover for to get the nodes and connections for the child
            if parent1.fitness > parent2.fitness:
                child_nodes, child_connections = rp.cross_over(parent1.genome, parent2.genome, rnd_child, config)
            else:
                child_nodes, child_connections = rp.cross_over(parent2.genome, parent1.genome, rnd_child, config)

            # Create child genome
            child_genome = Genome(child_seed, child_nodes, child_connections)

            # Mutate genome
            child_genome = rp.mutate_weights(child_genome, rnd_child, config)
            child_genome, _, _, _ = rp.mutate_add_node(child_genome, rnd_child, innovation_number_generator, config)
            child_genome, _ = rp.mutate_add_connection(child_genome, rnd_child, innovation_number_generator, config)

            child_agent = Agent(child_id, child_genome)
            new_agents.append(child_agent)

        # Select new representative
        existing_species = [ss.select_new_representative(species, rnd) for species in generation.species_list]
        # Reset members and fitness
        existing_species = [ss.reset_species(species) for species in existing_species]

        # Sort members into species
        new_species_list = ss.sort_agents_into_species(existing_species, new_agents, species_id_generator, config)

        # Filter out empty species
        new_species_list = ss.get_species_with_members(new_species_list)

        return Generation(generation.number + 1, new_generation_seed, new_agents, new_species_list)

        # new_agents = []
        # rnd = np.random.RandomState()
        # for agent in generation.agents:
        #     copied_genome = rp.deep_copy_genome(agent.genome)
        #     copied_genome = rp.mutate_weights(copied_genome, rnd, config)
        #     copied_genome = rp.mutate_bias(copied_genome, rnd, config)
        #     new_agents.append(Agent(1, copied_genome))

        # Clear species members

        # TODO sort agents into species, select new random representive

        # return Generation(generation.number + 1, 0, new_agents, [Species(1, new_agents[0].genome, new_agents)])

    def _cleanup(self, challenge: Challenge) -> None:
        # Notify callback and challenge
        challenge.clean_up()
        self.callback.on_cleanup()
