# Install MPI_NEAT
This project and all performance evaluation were done on multiple Raspberry Pi 4 with 4GB Ram each.
This guide describes how to set up and run the given program code.

## Setup Raspberry Pi
### General Setup
1. For this project the the RaspianBuster Image Version 2020-02-13 was used as OS. Download and install this image on all Raspberry Pis. Alternatively you can install all of the compnents on one Pi, create an .img from it, which can be flashed to all other devices. It is possible to use other operating systems, but on the raspberry pi it was then not possible to install tensorflow
2. Execute the following commands in the shell:
```shell script
# Update the system
sudo apt-get update
sudo apt-get upgrade

# Allow RDP connections
sudo apt-get install -y xrdp
```

### Create Python Environment

3. This project is developed with the python version 3.7.3, other versions were not tested. Ceate a virtual environment for this project.
```shell script
# Install pip && Virtual env package
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-venv

# Create a virutal environment
python3 -m venv neat_mpi_env
# Activate the newly created virutal environment
source ./neat_mpi_env/bin/activate
```

### Install TensorFlow

4. This project uses TensorFlow, which currently can't be installed with pip on ARM. So a custom .whl file is reqired. The installation for TensorFlow V.2.0.0. is done corresponding to this [guide](https://github.com/PINTO0309/Tensorflow-bin) with the .whl file taken from this [Github Repo](https://github.com/lhelontra/tensorflow-on-arm/releases/tag/v2.0.0)
```shell script
# Install some prelimanries
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran python-dev libgfortran5
sudo apt-get install -y libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev
sudo apt-get install -y liblapack-dev cython libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev

# Activate your python environment
pip install --upgrade setuptools
pip install keras_applications==1.0.8 --no-deps
pip install keras_preprocessing==1.1.0 --no-deps
pip install h5py==2.9.0
pip install pybind11
pip install -U --user six wheel mock

# For the Raspberry Pi 4, the armv7l version is required 
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.0.0/tensorflow-2.0.0-cp37-none-linux_armv7l.whl
pip install tensorflow-2.0.0-cp37-none-linux_armv7l.whl

# Test the installation
# Start the ptyhon interpreter
python
# Import tensorflow
import tensorflow as tf
tf.__version__
```

### Generate asymmetric keys for authentication
The nodes will communicate using ssh. To allow authentication password without a password, an asymmetric key must
be generated and added to the known hosts.
```shell script
#Login into your master node

# Install openssh-server, the device does not contain it already
sudo apt-get install -y openssh-server
# Generate an asymetric rsa key, skip the password
ssh-keygen -t rsa

# Login into your slave-node
# Make an .ssh directory, it not existing
mkdir .ssh

# Login into your master node

#T This will copy the public key to the authorized keys
cat .ssh/id_rsa.pub | ssh pi@ip_adress 'cat >> .ssh/authorized_keys'
```

### Install MPI & MPI4Py
Steps to intall the mpich, the MPI implementation and mpi4py for python.
```shell script
sudo apt-get install -y mpich
# Test if mpich is successfully installed with this command. THis should print the hostname 4 times (default raspberry)
mpiexec -n 4 hostname

# Activate the python environment and install mpi4py
source venv/neat_mpi_env/bin/activate
pip install mpi4py
 
#Execute a python script with
mpiexec -n numprocs python -m mpi4py scriptname.py
 ```

To use multiple machines/nodes, you have to create a machinefile or also called hostfile. This file contains all the ip 
addresses of the nodes. This repository contains already one, which should be edited accordingly.

After that you can run your mpi code on all machines. 
One important note: If you use mpi4py in a virtual environment, and start the script with the python command (as written
above), on the slave-nodes the default python environment will be used. In this case, there are not the dependencies 
installed. That's why the program must be started with the complete path to the python environment!
```shell script
#Execute a python script on multiple machines
mpiexec --hostfile /home/pi/mpi_neat/machinefile.txt -n 8 /home/pi/venv/mpi_test/bin/python3 -m mpi4py /home/pi/mpi_neat/src/mpi_test.py
```

### Upload to Code to the Raspberry Pi

3. Get the source code on the raspberry pi. TODO
4. Install the requirements 
```shell script
# Some preliminaries
sudo apt-get install libatlas-base-dev python3-dev

# Install the python requirements in your neat_mpi_env
# TODO
```

