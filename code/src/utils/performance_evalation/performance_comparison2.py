from __future__ import annotations

from typing import List

from utils.reporter.time_reporter import TimeReporterEntry


def speed_up_factor(single_time: float, multi_time: float) -> float:
    """
    Calculates the speed up factor between the given single and multi core time
    :param single_time: time of the single core run
    :param multi_time: time of the multi core run
    :return: the calculated speedup
    """
    return single_time / multi_time


def speed_up_abs(single_time: float, multi_time: float) -> float:
    """
    Calculates the absolute speedup between the single and multi core performance
    :param single_time: time of the single core run
    :param multi_time: time of the multi core run
    :return: the absolute speedup in seconds
    """
    return single_time - multi_time


def speed_up_ratio(single_time: float, multi_time: float, process_count: int) -> float:
    """
    Calculate the speedup ratio in percent between the single and multi core performance
    :param single_time: time of the single core run
    :param multi_time: time of the multi core run
    :param process_count: the amount of used processes
    :return: the speed up as percentage value
    """
    return speed_up_factor(single_time, multi_time) / process_count


def speed_up_values_all(single_core: List[TimeReporterEntry], multi_core: List[TimeReporterEntry],
                        process_counter: int) -> (float, float, float):
    sum_single = sum([e.reproduction_time for e in single_core])
    sum_single += sum([e.compose_offspring_time for e in single_core])
    sum_single += sum([e.evaluation_time for e in single_core])

    sum_multi = sum([e.reproduction_time for e in multi_core])
    sum_multi += sum([e.compose_offspring_time for e in multi_core])
    sum_multi += sum([e.evaluation_time for e in multi_core])

    abs_val = speed_up_abs(sum_single, sum_multi)
    factor_val = speed_up_factor(sum_single, sum_multi)
    ratio_val = speed_up_ratio(sum_single, sum_multi, process_counter)

    return abs_val, factor_val, ratio_val


def speed_up_value_reproduction_time(single_core: List[TimeReporterEntry], multi_core: List[TimeReporterEntry],
                                     process_counter: int) -> (float, float, float):
    sum_single = sum([e.reproduction_time for e in single_core])

    sum_multi = sum([e.reproduction_time for e in multi_core])

    abs_val = speed_up_abs(sum_single, sum_multi)
    factor_val = speed_up_factor(sum_single, sum_multi)
    ratio_val = speed_up_ratio(sum_single, sum_multi, process_counter)

    return abs_val, factor_val, ratio_val


def speed_up_value_compose_offspring(single_core: List[TimeReporterEntry], multi_core: List[TimeReporterEntry],
                                     process_counter: int) -> (float, float, float):
    sum_single = sum([e.compose_offspring_time for e in single_core])

    sum_multi = sum([e.compose_offspring_time for e in multi_core])

    abs_val = speed_up_abs(sum_single, sum_multi)
    factor_val = speed_up_factor(sum_single, sum_multi)
    ratio_val = speed_up_ratio(sum_single, sum_multi, process_counter)

    return abs_val, factor_val, ratio_val


def speed_up_value_evaluation_time(single_core: List[TimeReporterEntry], multi_core: List[TimeReporterEntry],
                                   process_counter: int) -> (float, float, float):
    sum_single = sum([e.evaluation_time for e in single_core])

    sum_multi = sum([e.evaluation_time for e in multi_core])

    abs_val = speed_up_abs(sum_single, sum_multi)
    factor_val = speed_up_factor(sum_single, sum_multi)
    ratio_val = speed_up_ratio(sum_single, sum_multi, process_counter)

    return abs_val, factor_val, ratio_val
