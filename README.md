# gpgpu-sim_UVM

Instructions to Set Up Simulation Environment

Install Ubuntu 16.04

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy python-six python3-six build-essential python-pip python3-pip python-virtualenv swig python-wheel python3-wheel libcurl3-dev libcupti-dev

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

Disable fast & safe boot in the bios and modify grub boot options to enable nomodeset [sudo nano /etc/default/grub and then GRUB_CMDLINE_LINUX_DEFAULT="nomodeset quiet splash"].

sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev
sudo apt-get install doxygen graphviz
sudo apt-get install python-pmw python-ply python-numpy libpng12-dev python-matplotlib
sudo apt-get install libxi-dev libxmu-dev freeglut3-dev

cd ~/Downloads
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
chmod +X cuda_8.0.61_375.26_linux-run
sudo ./cuda_8.0.61_375.26_linux-run [Note: NO to installing NVIDIA drivers and YES to creating symbolic link]

Install git, git-gui, ctags, cscope, libboost, vim
