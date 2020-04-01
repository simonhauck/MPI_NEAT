#!/home/pi/venv/mpi_test python3

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print('My rank is ',rank)
