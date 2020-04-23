from unittest import TestCase

from neat_core import activation_function as func


class ActivationFunctionTest(TestCase):

    def test_modified_sigmoid(self):
        self.assertEqual(0.5, func.modified_sigmoid(0))
        self.assertAlmostEqual(0.0073915, func.modified_sigmoid(-1), delta=0.0000001)
        self.assertAlmostEqual(0.9926084, func.modified_sigmoid(1), delta=0.0000001)
        self.assertAlmostEqual(0.7729422, func.modified_sigmoid(0.25), delta=0.0000001)
        self.assertAlmostEqual(0.2270577, func.modified_sigmoid(-0.25), delta=0.0000001)
