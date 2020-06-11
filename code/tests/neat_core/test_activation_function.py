from unittest import TestCase

from neat_core import activation_function as func


class ActivationFunctionTest(TestCase):

    def test_modified_sigmoid_activation(self):
        self.assertEqual(0.5, func.modified_sigmoid_activation(0))
        self.assertAlmostEqual(0.0073915, func.modified_sigmoid_activation(-1), delta=0.0000001)
        self.assertAlmostEqual(0.9926084, func.modified_sigmoid_activation(1), delta=0.0000001)
        self.assertAlmostEqual(0.7729422, func.modified_sigmoid_activation(0.25), delta=0.0000001)
        self.assertAlmostEqual(0.2270577, func.modified_sigmoid_activation(-0.25), delta=0.0000001)

    def test_step_activation(self):
        self.assertEqual(0, func.step_activation(-0.0001))
        self.assertEqual(0, func.step_activation(0))
        self.assertEqual(1, func.step_activation(2))

    def test_sigmoid_activation(self):
        self.assertAlmostEqual(0.5, func.sigmoid_activation(0))
        self.assertAlmostEqual(0.62245933120186, func.sigmoid_activation(0.5), delta=0.0000001)
        self.assertAlmostEqual(0.37754066879815, func.sigmoid_activation(-0.5), delta=0.0000001)

    def test_tanh_activation(self):
        self.assertAlmostEqual(0.761594155956, func.tanh_activation(1), delta=0.0000001)
        self.assertAlmostEqual(-0.761594155956, func.tanh_activation(-1), delta=0.0000001)
        self.assertAlmostEqual(0, func.tanh_activation(0), delta=0.0000001)
        self.assertAlmostEqual(0.46211715726, func.tanh_activation(0.5), delta=0.0000001)

    def test_relu_activation(self):
        self.assertAlmostEqual(1, func.relu_activation(1))
        self.assertAlmostEqual(0, func.relu_activation(0))
        self.assertAlmostEqual(0, func.relu_activation(-1))
