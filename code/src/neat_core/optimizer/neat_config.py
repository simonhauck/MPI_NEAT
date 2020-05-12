class NeatConfig(object):

    def __init__(self,
                 population_size: int = 150,
                 allow_recurrent_connections: bool = True,
                 connection_min_weight: float = -5.0,
                 connection_max_weight: float = 5.0,
                 probability_weight_mutation: float = 0.8,
                 probability_random_weight_mutation: float = 0.1,
                 weight_mutation_max_change: float = 1,
                 probability_mutate_add_connection: float = 0.05,
                 mutate_connection_tries=5,
                 probability_mutate_add_node: float = 0.03,
                 probability_enable_gene: float = 0.25
                 ) -> None:
        # General params
        self.population_size: int = population_size
        self.allow_recurrent: bool = allow_recurrent_connections

        # Connection params
        self.connection_min_weight: float = connection_min_weight
        self.connection_max_weight: float = connection_max_weight

        # Mutate weights param
        self.probability_weight_mutation: float = probability_weight_mutation
        self.probability_random_weight_mutation: float = probability_random_weight_mutation
        self.weight_mutation_max_change: float = weight_mutation_max_change

        # Mutate connection
        self.probability_mutate_add_connection: float = probability_mutate_add_connection
        self.mutate_connection_tries = mutate_connection_tries

        # Mutate node
        self.probability_mutate_add_node: float = probability_mutate_add_node

        # Crossover
        # If a connection is disabled in either parent, there is 75% chance that it is disabled, so 25% enabling chance
        self.probability_enable_gene: float = probability_enable_gene
