from unittest import TestCase

from neat_core.activation_function import modified_sigmoid_function
from neat_core.models.node import NodeType, Node


class NodeTest(TestCase):

    def test_node(self):
        input_node = Node(1, NodeType.INPUT, 0.3, modified_sigmoid_function, 0)
        output_node = Node(2, NodeType.OUTPUT, 0.4, modified_sigmoid_function, 1)
        hidden_node = Node("hidden_node", NodeType.HIDDEN, 0.6, modified_sigmoid_function, 0.5)

        self.assertEqual(1, input_node.innovation_number)
        self.assertEqual(NodeType.INPUT, input_node.node_type)
        self.assertEqual(0.3, input_node.bias)
        self.assertEqual(0.5, input_node.activation_function(0))
        self.assertEqual(0, input_node.x_position)

        self.assertEqual(2, output_node.innovation_number)
        self.assertEqual(NodeType.OUTPUT, output_node.node_type)
        self.assertEqual(0.4, output_node.bias)
        self.assertEqual(1, output_node.x_position)

        self.assertEqual("hidden_node", hidden_node.innovation_number)
        self.assertEqual(NodeType.HIDDEN, hidden_node.node_type)
        self.assertEqual(0.6, hidden_node.bias)
        self.assertEqual(0.5, hidden_node.x_position)
