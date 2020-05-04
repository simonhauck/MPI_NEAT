from neat_optimizer.neat_optimizer import NeatOptimizer


class NeatOptimizerSingleCore(NeatOptimizer):

    def create_next_generation(self):
        super().create_next_generation()
