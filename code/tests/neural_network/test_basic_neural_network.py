from unittest import TestCase

import numpy as np

from neat_core.activation_function import step_activation, modified_sigmoid_activation
from neat_core.models.connection import Connection
from neat_core.models.genome import Genome
from neat_core.models.node import Node, NodeType
from neural_network.basic_neural_network import BasicNeuralNetwork, BasicNeuron


class TestBasicNeuron(TestCase):

    def test_basic_neuron(self):
        weights = np.array([1.0, 2.0, 3.0])
        input_keys = np.array([10, 11, 12])

        neuron_valid = BasicNeuron(1, 1.5, weights, input_keys, step_activation, 1.0)
        self.assertEqual(1, neuron_valid.innovation_number)
        self.assertEqual(0, neuron_valid.val)
        self.assertEqual(0, neuron_valid.last_val)
        self.assertEqual(1, neuron_valid.x_position)
        self.assertEqual(1.5, neuron_valid.bias)
        self.assertFalse(neuron_valid.flag_calculated)
        self.assertTrue((weights == neuron_valid.weights).all())
        self.assertTrue((input_keys == neuron_valid.input_keys).all())

        with self.assertRaises(AssertionError):
            BasicNeuron(1, 1.1, weights, np.zeros([1]), step_activation, 1.0)


class TestBasicNeuralNetwork(TestCase):

    def setUp(self) -> None:
        self.feed_forward_hidden_bias = -0.5
        self.feed_forward_output_bias = -0.6

        self.nodes_feed_forward = [
            Node(1, NodeType.INPUT, 0, step_activation, x_position=0),
            Node(2, NodeType.INPUT, 0, step_activation, x_position=0),
            Node(3, NodeType.INPUT, 0, step_activation, x_position=0),
            Node(4, NodeType.OUTPUT, self.feed_forward_output_bias, step_activation, x_position=1),
            Node(15, NodeType.HIDDEN, self.feed_forward_hidden_bias, step_activation, x_position=0.5),
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

        self.genome_feed_forward = Genome(10, self.nodes_feed_forward, self.connections_feed_forward)
        # Truth table for the neural network above
        # X Y Z
        # Input 0 0 0: Result: 0.0
        # Input 0 0 1: Result: 1.0
        # Input 0 1 0: Result: 0.0
        # Input 0 1 1: Result: 1.0
        # Input 1 0 0: Result: 0.0
        # Input 1 0 1: Result: 1.0
        # Input 1 1 0: Result: 0.0
        # Input 1 1 1: Result: 1.0

        self.net_feed_forward = BasicNeuralNetwork()

        self.nodes_recurrent = [
            Node(1, NodeType.INPUT, 0, modified_sigmoid_activation, x_position=0),
            Node(2, NodeType.INPUT, 0, modified_sigmoid_activation, x_position=0),
            Node(3, NodeType.INPUT, 0, modified_sigmoid_activation, x_position=0),
            Node(4, NodeType.OUTPUT, -1.0, modified_sigmoid_activation, x_position=1),
            Node(10, NodeType.HIDDEN, -0.6, modified_sigmoid_activation, x_position=0.5),
            Node(15, NodeType.HIDDEN, -1.2, modified_sigmoid_activation, x_position=0.5),
        ]

        self.connections_recurrent = [
            Connection(innovation_number=11, input_node=1, output_node=10, weight=0.5, enabled=True),
            Connection(innovation_number=12, input_node=2, output_node=10, weight=-0.3, enabled=True),
            Connection(innovation_number=22, input_node=10, output_node=10, weight=1.5, enabled=True),
            Connection(innovation_number=21, input_node=15, output_node=10, weight=-0.1, enabled=True),
            Connection(innovation_number=16, input_node=2, output_node=15, weight=2.0, enabled=True),
            Connection(innovation_number=17, input_node=3, output_node=15, weight=-1.6, enabled=True),
            Connection(innovation_number=20, input_node=10, output_node=15, weight=-0.3, enabled=True),
            Connection(innovation_number=18, input_node=4, output_node=15, weight=0.6, enabled=True),
            Connection(innovation_number=13, input_node=10, output_node=4, weight=1.6, enabled=True),
            Connection(innovation_number=19, input_node=15, output_node=4, weight=-0.6, enabled=True),
            Connection(innovation_number=14, input_node=3, output_node=4, weight=-0.3, enabled=True),
        ]

        self.genome_recurrent = Genome(20, self.nodes_recurrent, self.connections_recurrent)

        self.net_recurrent = BasicNeuralNetwork()

    def test_build(self):
        neural_network = BasicNeuralNetwork()

        self.assertEqual(0, len(neural_network.input_neurons))
        self.assertEqual(0, len(neural_network.output_neurons))
        self.assertEqual(0, len(neural_network.all_neurons))
        self.assertEqual(0, len(neural_network.order))

        neural_network.build(self.genome_feed_forward)

        # Check input neurons
        self.assertEqual(3, len(neural_network.input_neurons))
        self.assertEqual(1, neural_network.input_neurons[0].innovation_number)
        self.assertEqual(2, neural_network.input_neurons[1].innovation_number)
        self.assertEqual(3, neural_network.input_neurons[2].innovation_number)
        for input_node in neural_network.input_neurons:
            self.assertEqual(input_node, neural_network.all_neurons[input_node.innovation_number])
            self.assertEqual(0, len(input_node.weights))
            self.assertEqual(0, len(input_node.input_keys))
            self.assertEqual(0, input_node.bias)

        # Check output neuron
        self.assertEqual(1, len(neural_network.output_neurons))
        # Check if output neuron is also in the all neurons dictionary
        self.assertEqual(neural_network.output_neurons[0],
                         neural_network.all_neurons[neural_network.output_neurons[0].innovation_number])
        self.assertEqual(4, neural_network.output_neurons[0].innovation_number)
        self.assertEqual(2, len(neural_network.output_neurons[0].weights))
        self.assertEqual(2, len(neural_network.output_neurons[0].input_keys))
        self.assertTrue((np.array([0.5, 2.0]) == neural_network.output_neurons[0].weights).all())
        self.assertTrue((np.array([1, 15]) == neural_network.output_neurons[0].input_keys).all())
        self.assertEqual(self.feed_forward_output_bias, neural_network.output_neurons[0].bias)

        # Check hidden neuron
        self.assertEqual(5, len(neural_network.all_neurons))
        self.assertIsNotNone(neural_network.all_neurons[15])
        self.assertEqual(3, len(neural_network.all_neurons[15].weights))
        self.assertEqual(3, len(neural_network.all_neurons[15].input_keys))
        self.assertTrue((np.array([-0.4, -1.0, 2.0]) == neural_network.all_neurons[15].weights).all())
        self.assertTrue((np.array([1, 2, 3]) == neural_network.all_neurons[15].input_keys).all())
        self.assertEqual(self.feed_forward_hidden_bias, neural_network.all_neurons[15].bias)

        # Check calculation order
        self.assertEqual([15, 4], neural_network.order)

    def test_reset(self):
        self.net_recurrent.build(self.genome_recurrent)

        input_val = [0.5, -2, 3]
        result_initially_expected = 0.03732211974054669
        result_second_expected = 0.1857531506810662

        # First activate
        result_first = self.net_recurrent.activate(input_val)
        self.assertAlmostEqual(result_initially_expected, result_first[0], delta=0.00000001)

        # Reset and test if calculation is the same
        self.net_recurrent.reset()
        result_first_reset = self.net_recurrent.activate(input_val)
        self.assertAlmostEqual(result_initially_expected, result_first_reset[0], delta=0.00000001)

        # Second calculation should now be different
        result_second = self.net_recurrent.activate(input_val)
        self.assertAlmostEqual(result_second_expected, result_second[0], delta=0.00000001)

    def test_activate(self):
        self.net_feed_forward.build(self.genome_feed_forward)

        with self.assertRaises(AssertionError):
            self.net_feed_forward.activate([1, 2, 3, 4])

        # Test input neurons
        self.net_feed_forward.activate([4, 5, 6])
        self.assertEqual(4, self.net_feed_forward.input_neurons[0].val)
        self.assertEqual(5, self.net_feed_forward.input_neurons[1].val)
        self.assertEqual(6, self.net_feed_forward.input_neurons[2].val)

        # Calculate results
        result_list_0_0_0 = self.net_feed_forward.activate([0, 0, 0])
        result_list_0_0_1 = self.net_feed_forward.activate([0, 0, 1])
        result_list_0_1_0 = self.net_feed_forward.activate([0, 1, 0])
        result_list_0_1_1 = self.net_feed_forward.activate([0, 1, 1])
        result_list_1_0_0 = self.net_feed_forward.activate([1, 0, 0])
        result_list_1_0_1 = self.net_feed_forward.activate([1, 0, 1])
        result_list_1_1_0 = self.net_feed_forward.activate([1, 1, 0])
        result_list_1_1_1 = self.net_feed_forward.activate([1, 1, 1])

        # Check result
        self.assertEqual([0], result_list_0_0_0)
        self.assertEqual([1], result_list_0_0_1)
        self.assertEqual([0], result_list_0_1_0)
        self.assertEqual([1], result_list_0_1_1)
        self.assertEqual([0], result_list_1_0_0)
        self.assertEqual([1], result_list_1_0_1)
        self.assertEqual([0], result_list_1_1_0)
        self.assertEqual([1], result_list_1_1_1)

    def test_activate_recurrent(self):
        self.net_recurrent.build(self.genome_recurrent)

        # First run Result with Input: 0.5, -2, 3 -> 0.03732211974054669
        # Second run Result with Input: 0.5, -2, 3 -> 0.1857531506810662

        # First activate
        result_first = self.net_recurrent.activate([0.5, -2, 3])
        self.assertEqual(1, len(result_first))
        self.assertAlmostEqual(0.03732211974054669, result_first[0], delta=0.00000001)

        # second activate
        second_results = self.net_recurrent.activate([0.5, -2, 3])
        self.assertEqual(1, len(second_results))
        self.assertAlmostEqual(0.1857531506810662, second_results[0], delta=0.00000001)

    def test_get_input_value_from_neuron(self):
        self.net_feed_forward.build(self.genome_feed_forward)

        # Set values
        self.net_feed_forward.all_neurons[15].val = 15
        self.net_feed_forward.all_neurons[15].last_val = 10

        # Feed forward with calculation_flag = False
        with self.assertRaises(ValueError):
            self.net_feed_forward._get_input_value_from_neuron(15, 1.0)

        # Recurrent connection with calculation_flag = False
        self.assertEqual(15, self.net_feed_forward._get_input_value_from_neuron(15, 0.5))
        self.assertEqual(15, self.net_feed_forward._get_input_value_from_neuron(15, 0))

        self.net_feed_forward.all_neurons[15].flag_calculated = True

        # Feed forward with calculation_flag = True
        self.assertEqual(15, self.net_feed_forward._get_input_value_from_neuron(15, 1.0))

        # Recurrent connection with calculation_flag = False
        self.assertEqual(10, self.net_feed_forward._get_input_value_from_neuron(15, 0.5))
        self.assertEqual(10, self.net_feed_forward._get_input_value_from_neuron(15, 0))

    def test_sort_connections(self):
        sorted_connections = self.net_feed_forward._sort_connections(self.connections_feed_forward)
        self.assertEqual(2, len(sorted_connections))

        # Test first key
        self.assertEqual(2, len(sorted_connections[4]))
        self.assertIn(self.connections_feed_forward[0], sorted_connections[4])
        self.assertIn(self.connections_feed_forward[2], sorted_connections[4])
        self.assertNotIn(self.connections_feed_forward[6], sorted_connections[4])

        # Test second connections
        self.assertEqual(3, len(sorted_connections[15]))
        self.assertIn(self.connections_feed_forward[1], sorted_connections[15])
        self.assertIn(self.connections_feed_forward[3], sorted_connections[15])
        self.assertIn(self.connections_feed_forward[5], sorted_connections[15])
