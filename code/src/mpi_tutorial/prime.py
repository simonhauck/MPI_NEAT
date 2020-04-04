import time
from loguru import logger
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

PRINT_NUMBERS = False
START_CHECKED_NUMBERS = 100000
END_CHECKED_NUMBERS = 110000


def is_prime(number: int) -> bool:
    if number <= 1:
        return False

    for i in range(2, number // 2):
        if number % i == 0:
            return False
    else:
        return True


def single_run_performance():
    primesfound = 0
    start_time = time.time()

    for i in range(START_CHECKED_NUMBERS, END_CHECKED_NUMBERS):
        is_prime(i)
        if is_prime(i):
            if PRINT_NUMBERS: logger.debug("Number " + str(i) + " is prime")
            primesfound = primesfound + 1

    end_time = time.time()
    total_time = end_time - start_time
    logger.info("All numbers checked. Total required time in seconds: " + str(total_time))
    return total_time, primesfound


END_TIME_SINGLE_CORE = 0
END_TIME_MULTI_CORE = 0

PRIMES_FOUND_SINGLE_CORE = 0
PRIMES_FOUND_MULTI_CORE = 0

if rank == 0:
    END_TIME_SINGLE_CORE, PRIMES_FOUND_SINGLE_CORE = single_run_performance()

    start_time = time.time()

    current_number = START_CHECKED_NUMBERS

    for i in range(1, size):
        comm.send(current_number, i)
        current_number = current_number + 1

    while True:
        result = comm.recv()

        # Check result
        if result[1]:
            if PRINT_NUMBERS: logger.debug("Number " + str(result[0]) + " is prime")
            PRIMES_FOUND_MULTI_CORE = PRIMES_FOUND_MULTI_CORE + 1

        # Last number received. print result
        if result[0] >= END_CHECKED_NUMBERS - 1:
            end_time = time.time()
            total_time = end_time - start_time
            logger.info("All numbers checked. Total required time in seconds: " + str(total_time))
            END_TIME_MULTI_CORE = total_time
            time.sleep(1)

            for i in range(1, size):
                comm.send(-1, i)

            break

        # Stop sending out new numbers
        if current_number < END_CHECKED_NUMBERS:
            comm.send(current_number, result[2])
            current_number = current_number + 1


else:
    while True:
        number_to_be_checked = comm.recv()

        # Cancel loop
        if number_to_be_checked == -1:
            logger.info("Node " + str(rank) + " with name " + name + " received stop signal")
            break
        result = is_prime(number_to_be_checked)
        comm.send([number_to_be_checked, result, rank], 0)

time.sleep(1)
if rank == 0:
    logger.info(
        "Primes found SingleCore: " + str(PRIMES_FOUND_SINGLE_CORE) + ", MultiCore: " + str(PRIMES_FOUND_MULTI_CORE))
    logger.info("Single Core Performance: " + str(END_TIME_SINGLE_CORE))
    logger.info("Multi Core Performance " + str(END_TIME_MULTI_CORE))
    logger.info("SpeedUp absolute: " + str(END_TIME_SINGLE_CORE - END_TIME_MULTI_CORE))
    logger.info("SpeedUp T(1)/T(N): " + str(END_TIME_SINGLE_CORE / END_TIME_MULTI_CORE))
    logger.info("SpeedUp Efficiency T(1)/T(N)/N: " + str(END_TIME_SINGLE_CORE / END_TIME_MULTI_CORE / size))
