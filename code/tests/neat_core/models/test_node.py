from unittest import TestCase

from neat_core.activation_function import modified_sigmoid
from neat_core.models.node import NodeType, Node


class NodeTest(TestCase):

    def test_node(self):
        input_node = Node(1, NodeType.INPUT, modified_sigmoid)
        output_node = Node(2, NodeType.OUTPUT, modified_sigmoid)
        hidden_node = Node(3, NodeType.HIDDEN, modified_sigmoid)

        self.assertEqual(1, input_node.id)
        self.assertEqual(NodeType.INPUT, input_node.node_type)
        self.assertEqual(0.5, input_node.activation_function(0))

        self.assertEqual(2, output_node.id)
        self.assertEqual(NodeType.OUTPUT, output_node.node_type)

        self.assertEqual(3, hidden_node.id)
        self.assertEqual(NodeType.HIDDEN, hidden_node.node_type)
