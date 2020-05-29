from typing import List

from .connection import Connection
from .node import Node


class Genome(object):

    def __init__(self, seed: int, nodes: List[Node] = [], connections: List[Connection] = []) -> None:
        """
        Create a genome, that encodes all information for a neural network
        :param seed:
        :param nodes:
        :param connections:
        """
        self.seed: int = seed
        self.nodes: List[Node] = nodes
        self.connections: List[Connection] = connections

        # Maybe use numpy random state to pass state between objects
        # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
