from abc import abstractmethod
from typing import List

from neat_core.models.generation import Generation
from neat_core.optimizer.neat_reporter import NeatReporter


class NeatOptimizerCallback(NeatReporter):

    @abstractmethod
    def on_finish(self, generation: Generation, reporters: List[NeatReporter]) -> None:
        """
        Called at the end of the evaluation process after cleanup.
        :param generation: the last generation evaluated
        :param reporters: a list with all reporters
        :return: None
        """
        pass

    @abstractmethod
    def finish_evaluation(self, generation: Generation) -> bool:
        """
        Return true, if the evaluation should be stopped
        :param generation: the generation that was evaluated last
        :return: true, if the evaluation should be stopped, false if a new generation should be built
        """
        pass
