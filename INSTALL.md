# Install MPI_NEAT
This project and all performance evaluation were done on multiple Raspberry Pis 4 with 4GB Ram each.
This guide describes how to set up and run the given program code.

## Setup Raspberry Pi
For this project the the RaspianBuster Image Version 2020-02-13 is used as OS. Download and install this image on one
Raspberry Pi. When all components are installed, an image will be created witch can be copied to the other devices. 
With this procedure, it is not required to perform all steps on all devices. Generally it is possible to use an other OS,
but the software was not tested and may require some changes in the configuration.
 
After the OS is installed execute the following commands in the shell:
```shell script
# Update the system
sudo apt-get update
sudo apt-get upgrade
```

### Create Python Environment
This project is developed with the python version 3.7.3, other versions were not tested. Create a virtual environment 
for this project.
```shell script
# Install pip && Virtual env package
sudo apt-get install -y python3-pip python3-venv

# Some preliminaries
sudo apt-get install -y libatlas-base-dev python3-dev

# Create a virtual environment
python3 -m venv neat_mpi_env
# Activate the newly created virtual environment
source ./neat_mpi_env/bin/activate
```

### Generate asymmetric keys for authentication
The nodes of the Beowulf Cluster will communicate using MPI, which requires SSH. To allow authentication without a password, an asymmetric key 
must be generated and added to the 'authorized_keys' file.
```shell script
# Install openssh-server, the device does not contain it already
sudo apt-get install -y openssh-server
# Generate an asymmetric rsa key, skip the password
ssh-keygen -t rsa
```
After generating the key the .ssh directory should contain an 'id_rsa' and 'id_rsa.pub' file. It is possible to
add the .pub key to the nodes 'authorized_keys' file manually. In this setup an different approach is used. The same 
private key is used for every node. This simplifies the setup. Add the generated key to the 'authorized_keys'. 
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

## Install required software
The following section shows which software is required to run this project and how it can be installed on the Raspberry
Pi.

### Install MPI & MPI4Py
This project was developed with MPICH and the python library mpi4py. The steps to install these components is described 
in the following section. 
```shell script
sudo apt-get install -y mpich libopenmpi-dev
# Test if mpich is successfully installed with this command. This should print the hostname 4 times (default raspberry)
mpiexec -n 4 hostname

# Activate the python environment and install mpi4py
source venv/neat_mpi_env/bin/activate
pip install mpi4py==3.0.3
 
#Execute a python script with
mpiexec -n NUMBER_OF_CORES python -m mpi4py PATH_TO_SCRIPT/mpi_hello_world.py
 ```
To use multiple machines/nodes, you have to create a machinefile or also called hostfile. This file contains all the ip 
addresses of the nodes. This repository contains this file already, which should be changed accord to your needs.

After that you can run your mpi code on all machines. One important note: If you use mpi4py in a virtual environment and 
start the script with the python command as described above, the default python environment will be used on the nodes. 
In this case, there are not the dependencies installed and the program will throw an error. That's why the program must 
be started with the complete path to the python environment!
```shell script
#Execute a python script on multiple machines
mpiexec --hostfile PATH_TO_MACHINEFILE -n NUMBER_OF_CORES PATH_TO_VENV/mpi_test/bin/python3 -m mpi4py PATH_TO_SCRIPT/mpi_hello_world.py
```

## Installation for Gym environments
Some environments require the following packages
```shell script
sudo apt-get install swig

pip install gym[box2d]
```

## Upload to Code to the Raspberry Pi and install the remaining dependencies
Last, upload the code to the Raspberry Pi. Activate the created virtual environment and install the remaining 
dependencies.
```shell script
# Activate the virtual environment
source ./neat_mpi_env/bin/activate

# Install the remaining dependencies
pip install -r requirements.txt
```
After this step, create an .img file and flash it to all other nodes. To run the code/tests, follow the instructions
in the Readme.md file. This image is called later base_image.

⚠️ IMPORTANT ⚠️: If you use Pycharm, it can be required to run the tests and the code that you mark the 'src' folder as
source. To to this, right click on the 'src' folder > Mark deployment as > Sources root

# Development
For active development and testing, a few additional steps can be helpful. These are introduced in the following 
sections.

## File synchronization
If you want to update the code, you can install NFS, which allows file synchronization. Use your stored base_image to 
create a master- and slave-node image. The setup is corresponding to this 
[guide](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/).

### Master node
To install the NFS-Server execute the following commands. This will create a shared_folder that can be accessed from the
slaves. If the program code is placed in this directory, the distribution and synchronisation is automatically handled.
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
Import the saved raspberry pi base_image. And execute the following commands to install the NFS-Client. The client
connects to the master with the ip 192.168.0.20 and imports the shared folder. You may have to change the up address.
```shell script
sudo apt-get install -y nfs-common

# Create the shared folder
mkdir shared_folder

# Open the config
sudo nano /etc/fstab
# Add the line, after that save and close (of course change the ip address, matching your setup)
192.168.0.20:/home/pi/shared_folder /home/pi/shared_folder nfs

# Mount files 
sudo mount -a

# Check the mounted folder with
df -h
# It should show something like "192.168.0.20:/home/pi/shared_folder   29G  3.1G   25G  11% /home/pi/shared_folder"
```
Reboot the system. After that, test the nfs by placing a file in the shared_folder on the master. If the file shows up 
in the client, everything works as intended. Create an image named image_slave and flash it on all nodes except the 
master node.

If it is not working after after the Raspberry Pi is booted, a bug can be the cause. The NFS is started before the 
network is ready. In this case you have to change the boot options on the Raspberry Pi, as shown in the following 
snipped. Alternatively you can execute the "sudo mount -a" command as shown above on every device after every reboot.
```shell script
sudo raspi-config

# Select 'Boot Options'
# Select 'Wait for Network at Boot'
```



