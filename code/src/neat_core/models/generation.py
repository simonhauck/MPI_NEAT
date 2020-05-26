from typing import List

from neat_core.models.agent import Agent
from neat_core.models.species import Species


class Generation(object):

    def __init__(self, number: int, seed: int, agents: List[Agent], species_list: List[Species]) -> None:
        self.number: int = number
        self.seed: int = seed
        self.species_list: List[Species] = species_list
        self.agents: List[Agent] = agents
