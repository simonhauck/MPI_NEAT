from unittest import TestCase
from utils.performance_evalation.stop_watch import StopWatch
from utils.performance_evalation import performance_comparison as pc


class PerformanceComparisonTest(TestCase):

    def setUp(self) -> None:
        self.speed_single_core = 12
        self.speed_multi_core = 3

        # Create two stopwatches
        self.single_stopwatch = StopWatch()
        self.single_stopwatch.total_stopped_time = 12
        self.multi_stopwatch = StopWatch()
        self.multi_stopwatch.total_stopped_time = 3

    def test_speed_up_factor(self):
        calculated_speedup = pc.speed_up_factor(self.single_stopwatch, self.multi_stopwatch)
        self.assertEqual(calculated_speedup, 4)

    def test_speed_up_abs(self):
        calculated_speedup = pc.speed_up_abs(self.single_stopwatch, self.multi_stopwatch)
        self.assertEqual(calculated_speedup, 9)

    def test_speed_up_ratio(self):
        calculated_speedup = pc.speed_up_ratio(self.single_stopwatch, self.multi_stopwatch, 8)
        self.assertEqual(calculated_speedup, 0.5)
