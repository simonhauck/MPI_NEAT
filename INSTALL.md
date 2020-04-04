# Install MPI_NEAT
This project and all performance evaluation were done on multiple Raspberry Pis 4 with 4GB Ram each.
This guide describes how to set up and run the given program code.

## Setup Raspberry Pi
### General Setup
For this project the the RaspianBuster Image Version 2020-02-13 was used as OS. Download and install this image on one
Raspberry Pi. When all components are installed, an image will be created witch can be copied to the other devices. 
With this procedure, it is not required to perform all steps on all devices.
It is possible to use other operating systems (like Ubuntu19), but on the raspberry pi it was then not 
possible to install TensorFlow.

After the OS is installed execute the following commands in the shell:
```shell script
# Update the system
sudo apt-get update
sudo apt-get upgrade

# Allow RDP connections
sudo apt-get install -y xrdp
```

### Create Python Environment
This project is developed with the python version 3.7.3, other versions were not tested. Create a virtual environment 
for this project.
```shell script
# Install pip && Virtual env package
sudo apt-get install -y python3-pip python3-venv

# Some preliminaries
sudo apt-get install -y libatlas-base-dev python3-dev

# Create a virutal environment
python3 -m venv neat_mpi_env
# Activate the newly created virutal environment
source ./neat_mpi_env/bin/activate
```

### Install TensorFlow
This project uses TensorFlow, which currently can't be installed with pip on ARM. So a custom .whl file is required. The 
installation for TensorFlow V.2.0.0. is done corresponding to this [guide](https://github.com/PINTO0309/Tensorflow-bin) 
with the .whl file taken from this [Github Repo](https://github.com/lhelontra/tensorflow-on-arm/releases/tag/v2.0.0)
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
# Activate your python environment before installing tensorflow
pip install tensorflow-2.0.0-cp37-none-linux_armv7l.whl
tf.__version__
```

The installation can be tested, with the following commands. You have to use your python environment.
```shell script
# Start the ptyhon interpreter
python
# Import tensorflow
import tensorflow as tf
tf.__version__
```

### Generate asymmetric keys for authentication
The nodes will communicate using mpi, which requires ssh. To allow authentication without a password, an asymmetric key 
must be generated and added to the 'authorized_keys' file.
```shell script
# Install openssh-server, the device does not contain it already
sudo apt-get install -y openssh-server
# Generate an asymetric rsa key, skip the password
ssh-keygen -t rsa
```
After generating the key the .ssh directory should contain an 'id_rsa' and 'id_rsa.pub' file. It is possible to
add the .pub key to the nodes 'authorized_keys' file manually. In this setup an different approach is used. The same 
private key is used for every node. This simplifies the setup. We add the generated key to the 'authorized_keys'. 
When the image is later be copied to all nodes, every node can connect every other.
```shell script
cd .ssh
# Add public key to authorized_keys
cat id_rsa.pub | cat >> authorized_keys
```
In the last preparation step, the host authentication fingerprint will be disabled for local connections.
Else you have to connect from every node to every node once to add the device to the 'known_hosts'.
```shell script
# Login into your master node
cd /etc/ssh/
sudo nano ssh_config

# Add the following before Host *
Host 192.168.0.*
   StrictHostKeyChecking no
```

### Install MPI & MPI4Py
Steps to install the mpich, the MPI implementation and mpi4py for python.
```shell script
sudo apt-get install -y mpich libopenmpi-dev
# Test if mpich is successfully installed with this command. This should print the hostname 4 times (default raspberry)
mpiexec -n 4 hostname

# Activate the python environment and install mpi4py
source venv/neat_mpi_env/bin/activate
pip install mpi4py==3.0.3
 
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
mpiexec --hostfile path_to_code/machinefile.txt -n 8 patz_to_venv/bin/python3 -m mpi4py path_to_code/mpi_test.py
```

### Upload to Code to the Raspberry Pi and install the remaining dependencies
Last, upload the code to the Raspberry Pi. Activate the created virtual environment and install the remaining 
dependencies. 
```shell script
# Activate the virutal environment
source ./neat_mpi_env/bin/activate

# Install the remaining dependencies
pip install -r requirements.txt
```
After this step, create an .img file and flash it to all other nodes. To run the code/tests, follow the instructions
in the Readme.md file. This image is called later base_image.

## File synchronization
If you want to update the code, you want to install NFS, which allows file synchronization. Use your stored image to 
create a master- and slave-node image. The setup is corresponding to this 
[guide](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/)

### Master node
Install the NFS-server
```shell script
sudo apt-get install -y nfs-kernel-server

# Create a shared folder, in this folder you have to place the code later
mkdir shared_folder

# OPen the nfs configfile
sudo nano /etc/exports
# Add the line, after that save and close
/home/pi/shared_folder *(rw,sync,no_root_squash,no_subtree_check)
# Save the config, after all changes
sudo exportfs -a
```
Create now an image and which is called image_master for the master node.

### Client node
Import the saved raspberry pi base_image. And execute the following commands
```shell script
sudo apt-get install -y nfs-common

# Create the shared folder
mkdir shared_folder

# Open the config
sudo nano /etc/fstab
# Add the line, after that save and close (ofcourse change the ip adress, matching your setup)
192.168.0.20:/home/pi/shared_folder /home/pi/shared_folder nfs

# Mount files 
sudo mount -a

# Check the mounted folder with
df -h
# It should show something like "192.168.0.20:/home/pi/shared_folder   29G  3.1G   25G  11% /home/pi/shared_folder"
```
Reboot the system. After that, test the nfs by placing a file in the shared_folder. If the file shows up in the client,
everything works as intended. Create an image named image_slave and flash it on all nodes except the master node.

If it is not working after boot, it is caused by a bug. The nfs is started before the network is ready, and then unable
to boot. On the Raspberry Pi this can be fixed by changing the boot option. Alternatively you can execute the
"sudo mount -a" command as shown above on every device after every reboot.
```shell script
sudo raspi-config

# Select 'Boot Options'
# Select 'Wait for Network at Boot'
```



