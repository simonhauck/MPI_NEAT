from unittest import TestCase

from neat_core.activation_function import step_function
from neat_core.models.connection import Connection
from neat_core.models.node import Node, NodeType
from neural_network.basic_neural_network import BasicNeuralNetwork


class TestBasicNeuralNetwork(TestCase):

    def setUp(self) -> None:
        nodes = [
            Node(1, NodeType.INPUT, step_function, 0),
            Node(2, NodeType.INPUT, step_function, 0),
            Node(3, NodeType.INPUT, step_function, 0),
            Node(4, NodeType.OUTPUT, step_function, 1),
            Node(15, NodeType.HIDDEN, step_function, 0.5),
        ]

        connections = [
            Connection(innovation_number=5, input_node=1, output_node=4, weight=0.5, enabled=True),
            Connection(innovation_number=16, input_node=1, output_node=15, weight=-0.4, enabled=True),
            Connection(innovation_number=17, input_node=15, output_node=4, weight=2.0, enabled=True),
            Connection(innovation_number=18, input_node=2, output_node=15, weight=-1.0, enabled=True),
            Connection(innovation_number=6, input_node=2, output_node=4, weight=-15.0, enabled=False),
            Connection(innovation_number=19, input_node=3, output_node=15, weight=2.0, enabled=True),
            Connection(innovation_number=19, input_node=3, output_node=4, weight=15.0, enabled=False)
        ]

        # Truth table for the neural network above
        # X Y Z
        # 0 0 0 -> 0
        # 0 0 1 -> 1
        # 0 1 0 -> 0
        # 0 1 1 -> 1
        # 1 0 0 -> 1
        # 1 1 0 -> 1
        # 1 1 1 -> 1

        self.basic_neural_net = BasicNeuralNetwork()

    def test_build(self):
        self.fail()

    def test_reset(self):
        self.fail()

    def test_activate(self):
        self.fail()

    def test_sort_connections(self):
        self.fail()
