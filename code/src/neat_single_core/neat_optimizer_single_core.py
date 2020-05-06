from neat_core.models.generation import Generation
from neat_core.models.genome import Genome
from neat_core.optimizer import Challenge
from neat_core.optimizer import NeatConfig
from neat_core.optimizer import NeatOptimizer


class NeatOptimizerSingleCore(NeatOptimizer):

    def start_evaluation(self, genome: Genome, challenge: Challenge, config: NeatConfig, seed: int = None) -> None:
        super().start_evaluation(genome, challenge, config, seed)

    def create_next_generation(self) -> Generation:
        return super().create_next_generation()

    def evaluate_generation(self) -> Generation:
        return super().evaluate_generation()

    def stop_evaluation(self) -> None:
        super().stop_evaluation()
