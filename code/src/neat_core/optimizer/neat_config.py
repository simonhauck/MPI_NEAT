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
                 compatibility_factor_disjoint_genes: float = 1.0,
                 compatibility_factor_matching_genes: float = 0.4,
                 compatibility_genome_size_threshold: int = 0,
                 compatibility_threshold: float = 3.0
                 ) -> None:
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

        # Compatibility factors
        self.compatibility_factor_disjoint_genes: float = compatibility_factor_disjoint_genes
        self.compatibility_factor_matching_genes: float = compatibility_factor_matching_genes
        self.compatibility_genome_size_threshold: int = compatibility_genome_size_threshold
        self.compatibility_threshold: float = compatibility_threshold
