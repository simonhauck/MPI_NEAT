# Install MPI_NEAT
This project and all performance evaluation were done on multiple Raspberry Pi 4 with 4GB Ram each.
This guide describes how to set up and run the given program code.

## Setup Raspberry Pi Image
1. For this project the 64bit Ubuntu 19.10 Version was used. Download and install this Image on all Raspberry Pis
2. Execute the following commands in the shell:
```shell script
sudo apt-get update
sudo apt-get upgrade

# To render the challenges
sudo apt-get install xubuntu-desktop

# Allow RDP connections
sudo apt-get install xrdp
sudo apt-get install xfce4
echo xfce4-session >~/.xsession
/etc/init.d/xrdp restart
```
3. This project was developed under the python version 3.7.5. Other versions were not tested
```shell script
# Install python, normally python is already included in Ubuntu 19
sudo apt-get install python3.7
# Install pip && Virtual env package
sudo apt-get install python3-pip
sudo apt-get install python3-venv

# Create a virutal environment
python3 -m venv env
# Activate the newly created virutal environment
source DIR/env/bin/activate
```

4. Install Pytorch on Raspberry Pi 4. There is no official release for the ARM architecture, so we use a wheel to install it. For further information, see this [Issue](https://github.com/simonhauck/MPI_NEAT/issues/17)
```shell script
# Get your specific architecture
uname -a
# With the Raspberry Pi 4 this should be something with aarch64. You need the corresponding wheel file. 

# Install some additional dependencies
sudo apt-get install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
# Get the .whl file. This repository contains the wheel file for the Raspberry Pi 4, which is originally taken from this [site](https://discuss.pytorch.org/t/pytorch-1-3-wheels-for-raspberry-pi-python-3-7/58580)

# not working currently
```

