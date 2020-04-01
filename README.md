# MPI_NEAT

![Python package Test](https://github.com/simonhauck/MPI_NEAT/workflows/Python%20package%20Test/badge.svg) 
[![codecov](https://codecov.io/gh/simonhauck/MPI_NEAT/branch/master/graph/badge.svg?token=8X3JMW3U9Z)](https://codecov.io/gh/simonhauck/MPI_NEAT)

## Run Code
```shell script
# On a single machine
# Activate the environment
source ./neat_mpi_env/bin/activate
mpiexec -n 4 python -m mpi4py /home/pi/mpi_neat/src/mpi_test.py

# Run on all cores
# In this case the environment is directly loaded
mpiexec --hostfile pathToMachineFile -n numberofCores /home/pi/venv/mpi_test/bin/python3 -m mpi4py /home/pi/mpi_neat/src/mpi_test.py
```

## Run UnitTests
1. Install the project according to the INSTALL.md file
2. To run the unit tests run the following code in the project directory:
```shell script
# Activate virutal env
source ./neat_mpi_env/bin/activate

# Run unit tests
pytest
```
3. To run the tests with code coverage run:
```shell script
# Activate virutal env
source ./neat_mpi_env/bin/activate

# Run tets with code coverage
pytest --cov=./code/src/
```

## Used Tools for the thesis:
- Visual Paradigm Community Edition 16.1 for the UML-Diagrams
- Latex for the Thesis
