from unittest import TestCase

from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import NodeType, Node
from utils.visualization.genome_visualization import _get_color_for_node_type, _sort_nodes_in_layers


class TestGenomeGraph(TestCase):

    def setUp(self) -> None:
        self.nodes_recurrent = [
            Node(1, NodeType.INPUT, 1.0, lambda x: x, x_position=0),
            Node(2, NodeType.INPUT, 1.2, lambda x: x, x_position=0),
            Node(3, NodeType.INPUT, 1.3, lambda x: x, x_position=0),
            Node(4, NodeType.OUTPUT, 1.4, lambda x: x, x_position=1),
            Node(10, NodeType.HIDDEN, 1.5, lambda x: x, x_position=0.5),
            Node(15, NodeType.HIDDEN, 1.6, lambda x: x, x_position=0.5),
        ]

        self.connections_recurrent = [
            Connection(innovation_number=11, input_node=1, output_node=10, weight=0.5, enabled=True),
            Connection(innovation_number=12, input_node=2, output_node=10, weight=-0.3, enabled=False),
            Connection(innovation_number=22, input_node=10, output_node=10, weight=1.5, enabled=True),
            Connection(innovation_number=21, input_node=15, output_node=10, weight=-0.1, enabled=False),
            Connection(innovation_number=16, input_node=2, output_node=15, weight=2.0, enabled=True),
            Connection(innovation_number=17, input_node=3, output_node=15, weight=-1.6, enabled=True),
            Connection(innovation_number=20, input_node=10, output_node=15, weight=-0.3, enabled=True),
            Connection(innovation_number=18, input_node=4, output_node=15, weight=0.6, enabled=True),
            Connection(innovation_number=13, input_node=10, output_node=4, weight=1.6, enabled=True),
            Connection(innovation_number=19, input_node=15, output_node=4, weight=-0.6, enabled=True),
            Connection(innovation_number=14, input_node=3, output_node=4, weight=-0.3, enabled=True),
        ]

        self.genome_recurrent = Genome(20, self.nodes_recurrent, self.connections_recurrent)

    def test_get_color_for_node_type(self):
        self.assertEqual('#ff9c45', _get_color_for_node_type(NodeType.INPUT))
        self.assertEqual('#4ada76', _get_color_for_node_type(NodeType.HIDDEN))
        self.assertEqual('#94bdff', _get_color_for_node_type(NodeType.OUTPUT))

    def test_sort_nodes_in_layers(self):
        sorted_nodes = _sort_nodes_in_layers(self.genome_recurrent)
        self.assertEqual(3, len(sorted_nodes))
        self.assertEqual([1, 2, 3], [node.innovation_number for node in sorted_nodes[0]])
        self.assertEqual([10, 15], [node.innovation_number for node in sorted_nodes[0.5]])
        self.assertEqual([4], [node.innovation_number for node in sorted_nodes[1]])
