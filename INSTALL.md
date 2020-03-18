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
sudo apt-get install  xfce4
echo xfce4-session >~/.xsession
/etc/init.d/xrdp restart
```
3. This project was developed under the python version 3.7.5. Other versions were not tested


