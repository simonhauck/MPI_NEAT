class NeatConfig(object):

    def __init__(self,
                 population_size: int = 150,
                 connection_min_weight: float = -5.0,
                 connection_max_weight: float = 5.0
                 ) -> None:
        # General params
        self.population_size: int = population_size

        # Connection params
        self.connection_min_weight: float = connection_min_weight
        self.connection_max_weight: float = connection_max_weight
