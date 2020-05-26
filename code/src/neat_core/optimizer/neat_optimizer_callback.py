from abc import ABC, abstractmethod

from neat_core.models.agent import Agent
from neat_core.models.generation import Generation


class NeatOptimizerCallback(ABC):

    def on_initialization(self) -> None:
        """
        Called before the first generation is evaluated. With the following generations, this method will not be invoked
        again.
        :return: None
        """
        pass

    def on_generation_evaluation_start(self, generation: Generation) -> None:
        """
        Called with the start of the evaluation of each generation.
        :param generation that will be evaluated
        :return: None
        """
        pass

    def on_agent_evaluation_start(self, i: int, agent: Agent) -> None:
        """
        Called with the start of the evaluation of each agent
        :param i a counter that increments for every agent
        :param agent the agent that will be evaluated
        :return: None
        """
        pass

    def on_agent_evaluation_end(self, i: int, agent: Agent) -> None:
        """
        Called with the end of the evaluation of each agent
        :param i a counter that increments for every agent
        :param agent the agent that will be evaluated
        :return: None
        """
        pass

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        """
        Called with the end of the evaluation of each generation
        :param generation the generation that was evaluated
        :return: None
        """
        pass

    def on_cleanup(self) -> None:
        """
        Called at the end of the evaluation process.
        :return: None
        """
        pass

    @abstractmethod
    def on_finish(self, generation: Generation) -> None:
        """
        Called at the end of the evaluation process after cleanup.
        :param generation: the last generation evaluated
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
