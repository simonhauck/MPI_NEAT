import os, sys, inspect

# Mpi cant be started in subdirectory, or to be more specific, can't import custom module
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from loguru import logger
from mpi4py import MPI

from utils.performance_evalation.stop_watch import StopWatch
from utils.performance_evalation.performance_comparison import speed_up_ratio, speed_up_abs, speed_up_factor

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

PRINT_NUMBERS = False
START_SEARCH = 1000000
END_SEARCH = 1001000


def is_prime(n):
    if n <= 1:
        return False

    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i = i + 6
    return True


def is_prime_slow(n):
    if n <= 1: return False

    # check for factors
    for i in range(2, n):
        if (n % i) == 0:
            return False
    else:
        return True


def single_core_solution(is_prime_function):
    amount_solutions = 0
    for i in range(START_SEARCH, END_SEARCH):
        if is_prime_function(i):
            amount_solutions += 1
            if PRINT_NUMBERS: logger.debug("{}. Found prime: {}".format(amount_solutions, i))

    return amount_solutions


# ----------------------------------------------------------------------------------------------------------------------
# Single core run
# ----------------------------------------------------------------------------------------------------------------------

USED_PRIME_FUNCTION = is_prime_slow

logger.debug("Node Info - Rank: {}, Name: {}, Size: {}".format(rank, name, size))
stopwatch_single_core = None
if rank == 0:
    logger.info("Starting single core run...")
    stopwatch_single_core = StopWatch(start=True)
    single_core_primes = single_core_solution(USED_PRIME_FUNCTION)
    stopwatch_single_core.stop()

    logger.info("Single core finished. Found Primes: {}, Required Time: {}s"
                .format(single_core_primes, stopwatch_single_core.total_stopped_time))

# ----------------------------------------------------------------------------------------------------------------------
# Multi core run
# ----------------------------------------------------------------------------------------------------------------------

# Startup:
number_to_send = START_SEARCH
amount_results_received = 0
amount_solutions = 0
stop_watch_multi_core = None
if rank == 0:
    logger.info("Starting multi core run...")
    stop_watch_multi_core = StopWatch(start=True)
    for i in range(1, size):
        comm.send(number_to_send, i)
        number_to_send += 1

# While
if rank == 0:
    while amount_results_received < END_SEARCH - START_SEARCH:
        result = comm.recv()
        amount_results_received += 1

        if result[0]:
            amount_solutions += 1
            if PRINT_NUMBERS: logger.debug("{}. Found prime: {}".format(amount_solutions, result[1]))

        if number_to_send < END_SEARCH:
            comm.send(number_to_send, result[2])
            number_to_send += 1
else:
    while True:
        number_to_check = comm.recv()
        # logger.debug("Rank {} received number {}".format(rank, number_to_check))
        if number_to_check < 0: break
        comm.send([USED_PRIME_FUNCTION(number_to_check), number_to_check, rank], 0)

# Cleanup
if rank == 0:
    stop_watch_multi_core.stop()
    logger.info("Multi core finished. Found Primes: {}, Required Time: {}s"
                .format(amount_solutions, stop_watch_multi_core.total_stopped_time))
    for i in range(1, size):
        comm.send(-1, i)

    # Evaluation
    logger.info("Evaluation Performance:")
    logger.info("Total speed up:    {}s".format(speed_up_abs(stopwatch_single_core, stop_watch_multi_core)))
    logger.info("Speed up factor:   {}".format(speed_up_factor(stopwatch_single_core, stop_watch_multi_core)))
    logger.info("Speed up ratio:    {}".format(speed_up_ratio(stopwatch_single_core, stop_watch_multi_core, size)))
