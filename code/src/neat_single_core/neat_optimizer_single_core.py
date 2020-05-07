from typing import Callable

from neat_core.optimizer.challenge import Challenge
from neat_core.optimizer.neat_config import NeatConfig
from neat_core.optimizer.neat_optimizer import NeatOptimizer


class NeatOptimizerSingleCore(NeatOptimizer):

    def start_evaluation(self, amount_input_nodes: int, amount_output_nodes,
                         activation_function: Callable[[float], float], challenge: Challenge, config: NeatConfig,
                         seed: int = None) -> None:
        super().start_evaluation(amount_input_nodes, amount_output_nodes, activation_function, challenge, config,
                                 seed)

    def evaluate_next_generation(self):
        super().evaluate_next_generation()

    def cleanup(self) -> None:
        super().cleanup()
