from loguru import logger
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#logger.debug("Rank: " + str(rank) + ", Size: " + str(size) + ", Name: " + name)


def shout_amount(number: int):
    logger.info("Rank {} on {} shouting: {}", rank, name, number)


def send_to_next(destination: int, data: int):
    comm.send(data, destination)


if rank == 0:
    logger.warning("All processes count UP!")
    amount = 1
    shout_amount(amount)
    send_to_next(rank + 1, amount)

    final_amount = comm.recv(source=size - 1)
    logger.warning("Counting complete: Reporting "+str(final_amount)+" of processes!")

else:
    data = comm.recv()
    newdata = data + 1
    shout_amount(newdata)
    send_to_next((rank + 1) % size, newdata)
