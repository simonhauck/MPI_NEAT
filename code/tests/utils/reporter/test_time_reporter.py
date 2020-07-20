import time
from unittest import TestCase

from neat_core.models.generation import Generation
from utils.reporter import time_reporter


class FitnessEvaluationUtilsTest(TestCase):

    def setUp(self) -> None:
        self.reporter = time_reporter.TimeReporter()
        self.generation = Generation(1, 1, [], [])

    def test_on_generation_evaluation_start(self):
        current_time = time.time()

        self.reporter.on_generation_evaluation_start(self.generation)
        self.assertAlmostEqual(current_time, self.reporter._tmp_on_generation_evaluation_start, delta=0.01)

    def test_on_generation_evaluation_end(self):
        self.assertEqual(0, len(self.reporter.data))

        self.reporter.on_generation_evaluation_start(self.generation)
        time.sleep(0.5)
        self.reporter.on_generation_evaluation_end(self.generation, [])

        # Check entry
        self.assertEqual(1, len(self.reporter.data))
        self.assertIn(self.reporter._tmp_entry, self.reporter.data)
        self.assertEqual(1, self.reporter._tmp_entry.generation)
        self.assertEqual(0, self.reporter._tmp_entry.reproduction_time)
        self.assertEqual(0, self.reporter._tmp_entry.compose_offspring_time)

        # Check time
        self.assertAlmostEqual(0.5, self.reporter._tmp_entry.evaluation_time, delta=0.01)

    def test_on_reproduction_start(self):
        # To create the temporary entry
        self.reporter.on_generation_evaluation_start(self.generation)
        self.reporter.on_generation_evaluation_end(self.generation, [])

        current_time = time.time()
        self.reporter.on_reproduction_start(self.generation)

        self.assertAlmostEqual(current_time, self.reporter._tmp_reproduction_start, delta=0.01)

    def test_on_compose_offsprings_start(self):
        # To create the temporary entry
        self.reporter.on_generation_evaluation_start(self.generation)
        self.reporter.on_generation_evaluation_end(self.generation, [])

        current_time = time.time()
        self.reporter.on_compose_offsprings_start()

        self.assertAlmostEqual(current_time, self.reporter._tmp_compose_offspring_start, delta=0.01)

    def test_on_compose_offsprings_end(self):
        # To create the temporary entry
        self.reporter.on_generation_evaluation_start(self.generation)
        self.reporter.on_generation_evaluation_end(self.generation, [])

        self.reporter.on_compose_offsprings_start()
        time.sleep(0.5)
        self.reporter.on_compose_offsprings_end()

        self.assertAlmostEqual(0.5, self.reporter._tmp_entry.compose_offspring_time, delta=0.01)

    def test_on_reproduction_end(self):
        # To create the temporary entry
        self.reporter.on_generation_evaluation_start(self.generation)
        self.reporter.on_generation_evaluation_end(self.generation, [])

        self.reporter.on_reproduction_start(self.generation)
        self.reporter.on_compose_offsprings_start()
        time.sleep(0.5)
        self.reporter.on_compose_offsprings_end()
        time.sleep(0.25)
        self.reporter.on_reproduction_end(self.generation)

        # Values are removed
        self.assertIsNone(self.reporter._tmp_entry)
        self.assertIsNone(self.reporter._tmp_on_generation_evaluation_start)
        self.assertIsNone(self.reporter._tmp_compose_offspring_start)
        self.assertIsNone(self.reporter._tmp_reproduction_start)

        # Check if list contains only one value
        self.assertEqual(1, len(self.reporter.data))
        entry = self.reporter.data[0]
        self.assertAlmostEqual(0.5, entry.compose_offspring_time, delta=0.01)
        self.assertAlmostEqual(0.25, entry.reproduction_time, delta=0.01)
