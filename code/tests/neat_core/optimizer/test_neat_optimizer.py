from unittest import TestCase

from neat_core.activation_function import step_function
from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType


# class InheritedNeatOptimizer(NeatOptimizer):
#
#     def start_evaluation(self, genome: Genome, config: NeatConfig, seed=None) -> None:
#         super().start_evaluation(genome, config, seed)


class TestNeatOptimizer(TestCase):

    def setUp(self) -> None:
        self.nodes_feed_forward = [
            Node(1, NodeType.INPUT, step_function, x_position=0),
            Node(2, NodeType.INPUT, step_function, x_position=0),
            Node(3, NodeType.INPUT, step_function, x_position=0),
            Node(4, NodeType.OUTPUT, step_function, x_position=1),
            Node(15, NodeType.HIDDEN, step_function, x_position=0.5),
        ]

        self.connections_feed_forward = [
            Connection(innovation_number=5, input_node=1, output_node=4, weight=0.5, enabled=True),
            Connection(innovation_number=16, input_node=1, output_node=15, weight=-0.4, enabled=True),
            Connection(innovation_number=17, input_node=15, output_node=4, weight=2.0, enabled=True),
            Connection(innovation_number=18, input_node=2, output_node=15, weight=-1.0, enabled=True),
            Connection(innovation_number=6, input_node=2, output_node=4, weight=-15.0, enabled=False),
            Connection(innovation_number=19, input_node=3, output_node=15, weight=2.0, enabled=True),
            Connection(innovation_number=19, input_node=3, output_node=4, weight=15.0, enabled=False)
        ]

        self.genome_feed_forward = Genome(1, 10, self.nodes_feed_forward, self.connections_feed_forward)

    def test_create_initial_generation(self):
        self.fail()
