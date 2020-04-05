import time


class StopWatch(object):

    def __init__(self, start: bool = False) -> None:
        """
        Create a stopwatch to track time. Can be used to track the execution time of given code pieces.
        The stop watch can be restarted, to accumulate multiple measurements
        :param start: true, if the watch should immediately track the time
        """
        self.total_stopped_time = 0
        self.start_time: float = None
        self.running = False
        if start:
            self.start()

    def start(self) -> float:
        """
        Start the stopwatch if it is not already running
        :return: the recorded start time
        """
        if not self.running:
            self.running = True
            self.start_time = time.time()
        return self.start_time

    def stop(self) -> float:
        """
        Stop the stopwatch and return the total stopped time.
        :return: the stopped time
        """
        if self.running:
            self.total_stopped_time += time.time() - self.start_time
            self.running = False
        return self.total_stopped_time
