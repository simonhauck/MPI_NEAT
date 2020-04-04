# MPI_NEAT

![Python package Test](https://github.com/simonhauck/MPI_NEAT/workflows/Python%20package%20Test/badge.svg) 
[![codecov](https://codecov.io/gh/simonhauck/MPI_NEAT/branch/master/graph/badge.svg?token=8X3JMW3U9Z)](https://codecov.io/gh/simonhauck/MPI_NEAT)

## Run Code
```shell script
# On a single machine
# Activate the environment
source ./neat_mpi_env/bin/activate
mpiexec -n NUMBER_OF_CORES python -m mpi4py PATH_TO_SCRIPT/mpi_hello_world.py
#Example
mpiexec -n 4 python -m mpi4py code/src/mpi_tutorial/mpi_hello_world.py

# Run on all cores
# In this case the environment is directly loaded
mpiexec --hostfile PATH_TO_MACHINEFILE -n NUMBER_OF_CORES PATH_TO_VENV/mpi_test/bin/python3 -m mpi4py PATH_TO_SCRIPT/mpi_hello_world.py
# Example for this setup
mpiexec --hostfile code/machinefile.txt -n 11 $HOME/venv/neat_mpi_env/bin/python3 -m mpi4py code/src/mpi_tutorial/mpi_hello_world.py
```

## Run UnitTests
1. Install the project according to the INSTALL.md file
2. To run the unit tests run the following code in the project directory:
```shell script
# Activate virutal env
source ./neat_mpi_env/bin/activate

# Go to the project directory

# Run unit tests
pytest
```
3. To run the tests with code coverage run:
```shell script
# Activate virutal env
source ./neat_mpi_env/bin/activate

# Go to the project directory

# Run tets with code coverage
pytest --cov=./code/src/
```

## Used Tools for the thesis:
- Visual Paradigm Community Edition 16.1 for the UML-Diagrams
- Latex for the Thesis
