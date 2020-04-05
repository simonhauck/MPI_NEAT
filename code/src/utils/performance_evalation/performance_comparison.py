from utils.performance_evalation.stop_watch import StopWatch


def speed_up_factor(single_stopwatch: StopWatch, multi_stopwatch: StopWatch) -> float:
    """
    Calculates the speed up factor between the given single and multi core time
    :param single_stopwatch: stopwatch with the measured single core performance
    :param multi_stopwatch: stopwatch with the measured multi core performance
    :return: the calculated speedup
    """
    return single_stopwatch.total_stopped_time / multi_stopwatch.total_stopped_time


def speed_up_abs(single_stopwatch: StopWatch, multi_stopwatch: StopWatch) -> float:
    """
    Calculates the absolute speedup between the single and multi core performance
    :param single_stopwatch: stopwatch with the measured single core performance
    :param multi_stopwatch: stopwatch with the measured multi core performance
    :return: the absolute speedup in seconds
    """
    return single_stopwatch.total_stopped_time - multi_stopwatch.total_stopped_time


def speed_up_ratio(single_stopwatch: StopWatch, multi_stopwatch: StopWatch, process_count: int) -> float:
    """
    Calculate the speedup ratio in percent between the single and multi core performance
    :param single_stopwatch: stopwatch with the measured single core performance
    :param multi_stopwatch: stopwatch with the measured multi core performance
    :param process_count: the amount of used processes
    :return: the speed up as percentage value
    """
    return speed_up_factor(single_stopwatch, multi_stopwatch) / process_count
