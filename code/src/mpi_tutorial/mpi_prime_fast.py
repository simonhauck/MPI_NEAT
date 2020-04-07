import os, sys, inspect

# Mpi cant be started in subdirectory, or to be more specific, can't import custom module
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from loguru import logger
from mpi4py import MPI

from utils.performance_evalation.stop_watch import StopWatch
from utils.performance_evalation.performance_comparison import speed_up_ratio, speed_up_abs, speed_up_factor

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
status = MPI.Status()

PRINT_NUMBERS = False
START_SEARCH = 1
END_SEARCH = 20000000


# 35.60801696777344s for 1 - 20011000 with 44cores, 1271256 primes found


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

USED_PRIME_FUNCTION = is_prime

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

STEP_SIZE = 11
# Startup: - First version
if rank == 0:
    stop_watch_multi_core = StopWatch(start=True)
else:
    stop_watch_multi_core = None

go_flag = comm.bcast(1, root=0)

i = START_SEARCH + rank * STEP_SIZE
found_primes_locally = 0
while i < END_SEARCH:
    # if i + STEP_SIZE > END_SEARCH:

    for j in range(i, i + STEP_SIZE):
        if USED_PRIME_FUNCTION(j):
            found_primes_locally += 1
    i += size * STEP_SIZE

# logger.debug("Rank {} finished calculating".format(rank))
found_primes_globally = comm.gather(found_primes_locally, root=0)

# Cleanup
if rank == 0:
    amount_solutions = sum(found_primes_globally)
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
