from unittest import TestCase

from neat_core.optimizer.neat_config import NeatConfig


class ConfigTest(TestCase):

    def test_config(self):
        config = NeatConfig(
            population_size=200,
            connection_min_weight=-10,
            connection_max_weight=10
        )

        self.assertEqual(200, config.population_size)
        self.assertEqual(-10, config.connection_min_weight)
        self.assertEqual(10, config.connection_max_weight)
