class NeatConfig(object):

    def __init__(self,
                 population_size: int = 150,
                 allow_recurrent_connections: bool = True,
                 connection_initial_min_weight: float = -3.0,
                 connection_initial_max_weight: float = 3.0,
                 connection_min_weight: float = -15.0,
                 connection_max_weight: float = 15.0,
                 bias_initial_min: float = -3.0,
                 bias_initial_max: float = 3.0,
                 bias_min: float = -15,
                 bias_max: float = 15,
                 probability_weight_mutation: float = 0.8,
                 probability_random_weight_mutation: float = 0.1,
                 weight_mutation_type: str = "normal",
                 weight_mutation_uniform_max_change: float = 1,
                 weight_mutation_normal_sigma: float = 1.3,
                 probability_bias_mutation: float = 0.8,
                 probability_random_bias_mutation: float = 0.1,
                 bias_mutation_type: str = "normal",
                 bias_mutation_uniform_max_change: float = 1,
                 bias_mutation_normal_sigma: float = 1.3,
                 probability_mutate_add_connection: float = 0.05,
                 mutate_connection_tries=5,
                 probability_mutate_add_node: float = 0.03,
                 probability_enable_gene: float = 0.25,
                 species_stagnant_after_generations: int = 15,
                 species_size_copy_best_genome: int = 5,
                 percentage_remove_low_genomes: float = 0.5,
                 compatibility_factor_disjoint_genes: float = 1.0,
                 compatibility_factor_matching_genes: float = 0.4,
                 compatibility_genome_size_threshold: int = 0,
                 compatibility_threshold: float = 3.0
                 ) -> None:
        """
        Create a config for the neat reproduction
        :param population_size: the population size of the generations
        :param allow_recurrent_connections: are recurrent connections allowed
        :param connection_initial_min_weight: the initial min weight of new connections
        :param connection_initial_max_weight: the initial max weight of new connections
        :param connection_min_weight: the min weight of connections that can be reached through mutation
        :param connection_max_weight: the max weight of connections that can be reached through mutation
        :param bias_initial_min: the initial min weight of a new bias
        :param bias_initial_max: the initial max weight of a new bias
        :param bias_min: the min value for a bias that can be reached through mutation
        :param bias_max: the max value for a bias that can be reached through mutation
        :param probability_weight_mutation: the probability that a weight of a connections is mutated
        :param probability_random_weight_mutation: the probability that a weight is assigned randomly and not perturbed
        :param weight_mutation_type: the type of distribution that is used for weigh mutation ("normal" or "uniform")
        :param weight_mutation_uniform_max_change: if type=uniform the max change of the weight (+- the value)
        :param weight_mutation_normal_sigma: if type=normal the sigma value of the normal distribution
        :param probability_bias_mutation: the probability of a bias mutation
        :param probability_random_bias_mutation: the probability that a random bias is assigned an not perturbed
        :param bias_mutation_type: the type of distribution that is used for bias mutation ("normal" or "uniform")
        :param bias_mutation_uniform_max_change: if type=uniform the max change of the bias (+- the value)
        :param bias_mutation_normal_sigma: if type=normal the sigma value of the normal distribution
        :param probability_mutate_add_connection: probability to add a new connection to a genome
        :param mutate_connection_tries: amount of tries to find a connection a new allowed connection
        :param probability_mutate_add_node: probability to add a node to a genome
        :param probability_enable_gene: probability to re-enable a gene, if it is disabled in both parents
        :param species_stagnant_after_generations: the amount of generations after which a species is stagnant
        :param species_size_copy_best_genome: the required size of species, that the best genome is copied
        :param percentage_remove_low_genomes: the percentage value of low genomes, that are removed before reproduction
        :param compatibility_factor_disjoint_genes: the factor for disjoint genes in the compatibility function
        :param compatibility_factor_matching_genes: the factor for matching genes in the compatibility function
        :param compatibility_genome_size_threshold: if genome size exceeds this value, the disjoint genes are normalized
        :param compatibility_threshold: the compatibility threshold, for two genomes to be in the same species
        """

        # General params
        self.population_size: int = population_size
        self.allow_recurrent: bool = allow_recurrent_connections

        # Connection params
        self.connection_initial_min_weight: float = connection_initial_min_weight
        self.connection_initial_max_weight: float = connection_initial_max_weight
        self.connection_min_weight: float = connection_min_weight
        self.connection_max_weight: float = connection_max_weight

        # Bias params
        self.bias_initial_min: float = bias_initial_min
        self.bias_initial_max: float = bias_initial_max
        self.bias_min: float = bias_min
        self.bias_max: float = bias_max

        # Mutate weights param
        self.probability_weight_mutation: float = probability_weight_mutation
        self.probability_random_weight_mutation: float = probability_random_weight_mutation
        self.weight_mutation_type: str = weight_mutation_type
        self.weight_mutation_uniform_max_change: float = weight_mutation_uniform_max_change
        self.weight_mutation_normal_sigma: float = weight_mutation_normal_sigma

        # Mutate bias param
        self.probability_bias_mutation: float = probability_bias_mutation
        self.probability_random_bias_mutation: float = probability_random_bias_mutation
        self.bias_mutation_type: str = bias_mutation_type
        self.bias_mutation_uniform_max_change: float = bias_mutation_uniform_max_change
        self.bias_mutation_normal_sigma: float = bias_mutation_normal_sigma

        # Mutate connection
        self.probability_mutate_add_connection: float = probability_mutate_add_connection
        self.mutate_connection_tries: int = mutate_connection_tries

        # Mutate node
        self.probability_mutate_add_node: float = probability_mutate_add_node

        # Crossover
        # If a connection is disabled in either parent, there is 75% chance that it is disabled, so 25% enabling chance
        self.probability_enable_gene: float = probability_enable_gene

        # Reproduction parameters
        self.species_stagnant_after_generations: int = species_stagnant_after_generations
        self.species_size_copy_best_genome: int = species_size_copy_best_genome
        self.percentage_remove_low_genomes: float = percentage_remove_low_genomes

        # Compatibility factors
        self.compatibility_factor_disjoint_genes: float = compatibility_factor_disjoint_genes
        self.compatibility_factor_matching_genes: float = compatibility_factor_matching_genes
        self.compatibility_genome_size_threshold: int = compatibility_genome_size_threshold
        self.compatibility_threshold: float = compatibility_threshold
