from typing import List

from neat_core.models.agent import Agent
from neat_core.models.generation import Generation
from neat_core.models.species import Species


class NeatOptimizerCallback(object):

    def on_initialization(self) -> None:
        """
        Called before the first generation is evaluated. With the following generations, this method will not be invoked
        again.
        :return: None
        """
        pass

    def on_generation_evaluation_start(self, generation: Generation, species_list: List[Species]) -> None:
        """
        Called with the start of the evaluation of each generation.
        :return: None
        """
        pass

    def on_agent_evaluation_start(self, agent: Agent) -> None:
        """
        Called with the start of the evaluation of each agent
        :return: None
        """
        pass

    def on_agent_evaluation_end(self, agent: Agent) -> None:
        """
        Called with the end of the evaluation of each agent
        :return: None
        """
        pass

    def on_generation_evaluation_end(self, generation: Generation, species_list: List[Species]) -> None:
        """
        Called with the end of the evaluation of each generation
        :return: None
        """
        pass

    def on_cleanup(self) -> None:
        """
        Called at the end of the evaluation process.
        :return: None
        """
        pass
