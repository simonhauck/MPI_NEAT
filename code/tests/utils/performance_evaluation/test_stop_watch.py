from unittest import TestCase
from utils.performance_evalation.stop_watch import StopWatch
import time


class StopWatchTest(TestCase):

    def test_init(self):
        default_watch = StopWatch()
        self.assertEqual(default_watch.total_stopped_time, 0)
        self.assertEqual(default_watch.running, False)
        self.assertIsNone(default_watch.start_time)

        running_watch = StopWatch(start=True)
        self.assertEqual(running_watch.total_stopped_time, 0)
        self.assertEqual(running_watch.running, True)
        self.assertAlmostEqual(running_watch.start_time, time.time(), delta=0.0001,
                               msg="The start time of the stopwatch is not matching the expected value")

    def test_start(self):
        default_watch = StopWatch()

        # Starting watch first time
        start_time = default_watch.start()
        self.assertAlmostEqual(start_time, time.time(), delta=0.0001)
        self.assertAlmostEqual(default_watch.start_time, time.time(), delta=0.0001)
        self.assertEqual(default_watch.running, True)

        time.sleep(0.1)

        # Starting watch second time
        default_watch.start()
        self.assertEqual(default_watch.start_time, start_time)

    def test_stop(self):
        default_watch = StopWatch()
        start_time = default_watch.start()
        time.sleep(0.1)

        measured_time = default_watch.stop()
        stopped_at = time.time()

        self.assertAlmostEqual(measured_time, stopped_at - start_time, delta=0.0001,
                               msg="The measured time did not match the expected value")

        # Restart the watch
        start_time2 = default_watch.start()
        time.sleep(0.1)
        measured_time = default_watch.stop()
        stopped_at2 = time.time()
        self.assertAlmostEqual(measured_time, stopped_at - start_time + stopped_at2 - start_time2, delta=0.0001,
                               msg="The measured time did not match the expected value")
