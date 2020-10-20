# MPI_NEAT

![Python package Test](https://github.com/simonhauck/MPI_NEAT/workflows/Python%20package%20Test/badge.svg) 
[![codecov](https://codecov.io/gh/simonhauck/MPI_NEAT/branch/master/graph/badge.svg?token=8X3JMW3U9Z)](https://codecov.io/gh/simonhauck/MPI_NEAT)

This repository was created as part of my master thesis. The goal was to implement and analyze the [NEAT](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) algorithm and
parallelize the execution with MPI on a Beowulf cluster. This repository contains the program code as well as the actual thesis and images.
If you are interested in this project this README gives an overview over the results and how to run the code. For a setup guide
follow the instructions in the INSTALL.md file.

## How to run this project
In my thesis I used a cluster of 10 Raspberry Pi 4s to execute the program code. But it is also possible to use a single 
machine with multiple cores. This implementation has in contrast to the standard python interpreter the advantage, that 
the complete CPU can be utilized. 

### Unit Tests
Before you execute the project you can check with the unit test that everything works as intended. To run all tests run the 
following command in the project directory after activating the virtual environment:
```shell script
# Run unit tests
pytest
```
To measure the code coverage run the following command. Note that only the NEAT functions are tested and not the MPI 
implementation (This is the reason for the low code coverage). Unit testing MPI code is much more difficult and 
therefore not part of this project.
```shell script
# Run unit test with code coverage
pytest --cov-config=code/src/.coveragerc --cov=code/src/
```
If you want to test your setup and check if everything regarding MPI works as intended, feel free to use my hello world script.
Every node prints a hello world string with its rank and the device name. You can execute the program with the following
command.
```shell script
mpiexec -n 4 python -m mpi4py code/src/mpi_tutorial/mpi_hello_world.py
```
The parameter "-n" indicates the amount of processes to be used. In this example four processes are used and they are all
started on the same machine. To utilize the complete cluster a machinefile or hostfile is required, where the ip addresses
of all nodes are specified. This project contains already a machine file which must be modified. To execute the the 
hello world program use the following command
```shell script
mpiexec --machinefile code/machinefile.txt -n 40 $HOME/venv/neat_mpi_env/bin/python3 -m mpi4py code/src/mpi_tutorial/mpi_hello_world.py
```
The parameter "--machinefile" contains the path to the machinefile. In addition is the path to the python interpreter given.
This is required, because the virtual environment can't be activated on the nodes. As you can see, in this case are forty
processes used.

### Start a Training process
The project contains two scripts which are used to start the training/optimization process and to visualize the results.
First the script to train a model is introduced. This project contains a few examples, which are in the
examples folder. Generally, all examples can be trained with a sequential or a parallel algorithm. With the same seed, 
both implementations produce the same results. To start a training process with the sequential algorithm use the following
command with the activated virtual environment.
```shell script
python code/src/main.py xor -s 1
```
The parameter "-s" indicates the seed for the environment. With the same seed, the same results are generated. To run
the same example with the MPI implementation, use the following command. Note that the current directory must be the 
"src" folder.
```shell script
mpiexec --machinefile ../machinefile.txt -n 40 $HOME/venv/neat_mpi_env/bin/python3 -m mpi4py.futures main.py xor -s 1 -o mpi
```



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
pytest --cov-config=code/src/.coveragerc --cov=code/src/
```

## Run Examples
The following commands star the corresponding examples
```shell script
# Activate virutal env
source ./neat_mpi_env/bin/activate

# Go to the project directory

# The xor problem
python code/src/main.py xor

# The mountain car challenge from the open ai gym
python code/src/main.py mountain_car
```

## Used Tools for the thesis:
- Visual Paradigm Community Edition 16.1 for the UML-Diagrams
- Latex for the Thesis
