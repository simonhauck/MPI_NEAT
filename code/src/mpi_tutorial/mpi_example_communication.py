import os, sys, inspect

# Mpi cant be started in subdirectory, or to be more specific, can't import custom module
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from loguru import logger
from mpi4py import MPI
import time
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
status = MPI.Status()

logger.debug("NodeInfo - Rank: {}, Size: {}, Name: {}".format(rank, size, name))

time.sleep(0.2)
if rank == 0: logger.info("----- Point-to-point communication example -----")

target_node = 1
if rank == 0:
    data_to_send = 23
    comm.send(data_to_send, dest=1)
    logger.debug("Send data {} to node {}".format(data_to_send, target_node))

if rank == target_node:
    data_recv = comm.recv()
    logger.debug("Data {} received on node {}".format(data_recv, target_node))

if rank == 0: logger.info("----- Point-to-point communication example with tags-----")
time.sleep(0.5)

target_node = 1
if rank == 0:
    first_tag_msg = [1, 2, 3]
    second_tag_msg = [6, 7, 8]
    comm.send(first_tag_msg, dest=target_node, tag=1)
    logger.debug("Send first message {} to target_node {} with tag 1".format(first_tag_msg, target_node))
    comm.send(second_tag_msg, dest=target_node, tag=2)
    logger.debug("Send second message {} to target_node {} with tag 2".format(second_tag_msg, target_node))

if rank == target_node:
    second_tag_msg_recv = comm.recv(tag=2, status=status)
    logger.debug("Received message with tag 2 and data {}".format(second_tag_msg_recv))
    logger.debug("Status info -  source: {}, tag: {}".format(status.source, status.tag))
    first_tag_msg_recv = comm.recv(tag=1, status=status)
    logger.debug("Received message with tag 1 and data {}".format(first_tag_msg_recv))

if rank == 0: logger.info("----- Broadcast communication example with tags-----")
time.sleep(0.5)

if rank == 0:
    bcast_data = "Hello, this is a broadcast message"
else:
    bcast_data = None

received_bcas_data = comm.bcast(bcast_data, root=0)
logger.debug("Rank {} received broadcast data: {}".format(rank, received_bcas_data))

if rank == 0: logger.info("----- Scatter communication -----")
time.sleep(0.5)
if rank == 0:
    data = [random.randint(0, 10) for _ in range(size)]
    logger.debug("Master generated data: {}".format(data))
else:
    data = None

scattered_data = comm.scatter(data, root=0)
logger.debug("Process rank {} received data: {}".format(rank, scattered_data))

if rank == 0: logger.info("----- Gather communication -----")
time.sleep(0.5)
scattered_data *= -1

gathered_data = comm.gather(scattered_data, root=0)

if rank == 0:
    logger.debug("Gathered data: {}".format(gathered_data))
