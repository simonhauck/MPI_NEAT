from unittest import TestCase

from neat_core import activation_function as func


class ActivationFunctionTest(TestCase):

    def test_modified_sigmoid(self):
        self.assertEqual(0.5, func.modified_sigmoid_function(0))
        self.assertAlmostEqual(0.0073915, func.modified_sigmoid_function(-1), delta=0.0000001)
        self.assertAlmostEqual(0.9926084, func.modified_sigmoid_function(1), delta=0.0000001)
        self.assertAlmostEqual(0.7729422, func.modified_sigmoid_function(0.25), delta=0.0000001)
        self.assertAlmostEqual(0.2270577, func.modified_sigmoid_function(-0.25), delta=0.0000001)

    def test_step_function(self):
        self.assertEqual(0, func.step_function(-0.0001))
        self.assertEqual(0, func.step_function(0))
        self.assertEqual(1, func.step_function(2))
