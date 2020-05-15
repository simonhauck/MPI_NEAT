from unittest import TestCase

from neat_core.activation_function import step_function
from neat_core.models.node import Node, NodeType
from neat_single_core.inno_number_generator_single_core import InnovationNumberGeneratorSingleCore


class InnovationNumberGeneratorSingleCoreTest(TestCase):

    def setUp(self) -> None:
        self.generator = InnovationNumberGeneratorSingleCore()

    def test_next_generation(self):
        node1 = Node(1, NodeType.INPUT, 1.0, step_function, 0)
        node2 = Node("abc", NodeType.HIDDEN, 1.1, step_function, 0.5)
        node3 = Node(3, NodeType.OUTPUT, 1.2, step_function, 1)

        generator = InnovationNumberGeneratorSingleCore()
        # Nodes
        self.assertEqual(0, generator.get_node_innovation_number())
        self.assertEqual(1, generator.get_node_innovation_number(node1, node2))
        self.assertEqual(1, generator.get_node_innovation_number(node1, node2))

        # Connections
        self.assertEqual(0, generator.get_connection_innovation_number())
        self.assertEqual(1, generator.get_connection_innovation_number(node1, node3))
        self.assertEqual(1, generator.get_connection_innovation_number(node1, node3))

        generator.next_generation(5)

        # Nodes
        self.assertEqual(2, generator.get_node_innovation_number(node1, node2))
        self.assertEqual(2, generator.get_node_innovation_number(node1, node2))

        # Connections
        self.assertEqual(2, generator.get_connection_innovation_number())
        self.assertEqual(1, generator.get_connection_innovation_number(node1, node3))
        self.assertEqual(3, generator.get_connection_innovation_number(node2, node3))
        self.assertEqual(3, generator.get_connection_innovation_number(node2, node3))

    def test_get_node_innovation_number(self):
        node1 = Node(1, NodeType.INPUT, 1.0, step_function, 0)
        node2 = Node("abc", NodeType.HIDDEN, 1.1, step_function, 0.5)
        node3 = Node(3, NodeType.OUTPUT, 1.2, step_function, 1)

        generator = InnovationNumberGeneratorSingleCore()
        self.assertEqual(0, generator.get_node_innovation_number())
        self.assertEqual(1, generator.get_node_innovation_number())
        self.assertEqual(2, generator.get_node_innovation_number(node1, node2))
        self.assertEqual(2, generator.get_node_innovation_number(node1, node2))
        self.assertEqual(3, generator.get_node_innovation_number(node1, node3))
        self.assertEqual(4, generator.get_node_innovation_number())
        self.assertEqual(3, generator.get_node_innovation_number(node1, node3))
        self.assertEqual(5, generator.get_node_innovation_number(node3, node1))

        self.assertEqual(0, generator.get_connection_innovation_number())

    def test_get_connection_innovation_number(self):
        node1 = Node(1, NodeType.INPUT, 1.0, step_function, 0)
        node2 = Node("abc", NodeType.HIDDEN, 1.1, step_function, 0.5)
        node3 = Node(3, NodeType.OUTPUT, 1.2, step_function, 1)

        generator = InnovationNumberGeneratorSingleCore()
        self.assertEqual(0, generator.get_connection_innovation_number())
        self.assertEqual(1, generator.get_connection_innovation_number())
        self.assertEqual(2, generator.get_connection_innovation_number(node1, node3))
        self.assertEqual(2, generator.get_connection_innovation_number(node1, node3))
        self.assertEqual(3, generator.get_connection_innovation_number())
        self.assertEqual(4, generator.get_connection_innovation_number(node2, node3))
        self.assertEqual(4, generator.get_connection_innovation_number(node2, node3))
        self.assertEqual(2, generator.get_connection_innovation_number(node1, node3))
        self.assertEqual(5, generator.get_connection_innovation_number(node3, node1))

        self.assertEqual(0, generator.get_node_innovation_number())
