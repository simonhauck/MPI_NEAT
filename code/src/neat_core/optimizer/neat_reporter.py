from abc import ABC

from neat_core.models.agent import Agent
from neat_core.models.generation import Generation


class NeatReporter(ABC):

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

    def on_finish(self, generation: Generation) -> None:
        """
        Called at the end of the evaluation process after cleanup.
        :param generation: the last generation evaluated
        :return: None
        """
        pass

    def on_reproduction_start(self, generation: Generation) -> None:
        """
        Called at the start of the reproduction functions.
        This includes cleaning out the species, assigning offspring for each species, building parent pairs, the
        crossover and mutation process as well as sorting the agents into species
        :param generation: the old generation
        :return: None
        """
        pass

    def on_compose_offsprings_start(self) -> None:
        """
        Called before the offsprings are composed
        :return: None
        """
        pass

    def on_compose_offsprings_end(self) -> None:
        """
        Called when the composing offspring process is finished.
        This includes, the the crossover and mutation process
        :return: None
        """
        pass

    def on_reproduction_end(self, generation: Generation) -> None:
        """
        Called at the end of the reproductions
        :param generation:
        :return:
        """
        pass
