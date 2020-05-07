from typing import List

from neat_core.models.agent import Agent
from neat_core.models.species import Species


class Generation(object):

    def __init__(self, generation: int, agents: List[Agent], species_list: List[Species]) -> None:
        self.number: int = generation
        self.species_list: List[Species] = species_list
        self.agents: List[Agent] = agents
