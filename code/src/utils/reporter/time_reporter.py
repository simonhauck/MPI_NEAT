import time
from typing import List

from neat_core.models.generation import Generation
from neat_core.optimizer.neat_reporter import NeatReporter


class TimeReporterEntry(object):

    def __init__(self, generation: int, reproduction_time: float = 0, compose_offspring_time: float = 0,
                 evaluation_time: float = 0) -> None:
        """
        Create an entry for the TimeReporter
        :param generation: the generation in which the values were measured
        :param reproduction_time: the time for the reproduction, without the compose offspring time
        :param compose_offspring_time: the time for the compose offspring
        :param evaluation_time: the time for the evaluation
        """
        self.generation = generation
        self.reproduction_time = reproduction_time
        self.compose_offspring_time = compose_offspring_time
        self.evaluation_time = evaluation_time


class TimeReporter(NeatReporter):

    def __init__(self) -> None:
        self.data: List[TimeReporterEntry] = []

        self._tmp_reproduction_start = None
        self._tmp_compose_offspring_start = None
        self._tmp_on_generation_evaluation_start = None
        self._tmp_entry: TimeReporterEntry = None

    def on_generation_evaluation_start(self, generation: Generation) -> None:
        """
        Store a timestamp when the evaluation of agents starts
        :param generation: not used
        :return: None
        """
        assert self._tmp_on_generation_evaluation_start is None
        assert self._tmp_entry is None
        self._tmp_on_generation_evaluation_start = time.time()

    def on_generation_evaluation_end(self, generation: Generation) -> None:
        """
        Calculate the required time for evaluating all agents, create a new entry with the required times and store
        it in a list. Additionally the values are stored in a temporary variable
        :param generation: the current generation
        :return: None
        """
        assert self._tmp_on_generation_evaluation_start is not None
        assert self._tmp_entry is None

        required_time = time.time() - self._tmp_on_generation_evaluation_start
        self._tmp_entry = TimeReporterEntry(generation.number, evaluation_time=required_time)
        self.data.append(self._tmp_entry)

    def on_reproduction_start(self, generation: Generation) -> None:
        """
        Store the timestamp when the reproduction started
        :param generation: not used
        :return: None
        """
        assert self._tmp_reproduction_start is None
        assert self._tmp_entry is not None
        self._tmp_reproduction_start = time.time()

    def on_compose_offsprings_start(self) -> None:
        """
        Store the timestamp when the composing of offsprings starts
        :return: None
        """
        assert self._tmp_compose_offspring_start is None
        assert self._tmp_entry is not None
        self._tmp_compose_offspring_start = time.time()

    def on_compose_offsprings_end(self) -> None:
        """
        Calculate the required time to compose the offsprings
        :return: None
        """
        assert self._tmp_compose_offspring_start is not None
        assert self._tmp_entry is not None
        self._tmp_entry.compose_offspring_time = time.time() - self._tmp_compose_offspring_start

    def on_reproduction_end(self, generation: Generation) -> None:
        """
        Calculate the required time for reproduction.
        This is calculated by subtracting the start time and the compose off spring time from the current time.
        Lastly the values the temporary values are reset
        :param generation: not used
        :return: None
        """
        assert self._tmp_reproduction_start is not None
        assert self._tmp_entry is not None
        required_time = time.time() - self._tmp_reproduction_start - self._tmp_entry.compose_offspring_time
        self._tmp_entry.reproduction_time = required_time

        self._tmp_entry = None
        self._tmp_reproduction_start = None
        self._tmp_on_generation_evaluation_start = None
        self._tmp_compose_offspring_start = None
