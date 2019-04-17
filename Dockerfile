FROM ubuntu:16.04
MAINTAINER Debashis Ganguly (debashis@cs.pitt.edu)
RUN apt update
RUN apt upgrade -y
RUN apt install -y openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy python-six python3-six build-essential python-pip python3-pip python-virtualenv swig python-wheel python3-wheel libcurl3-dev libcupti-dev software-properties-common
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt update
RUN apt install -y build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev doxygen graphviz python-pmw python-ply python-numpy libpng12-dev python-matplotlib libxi-dev libxmu-dev freeglut3-dev wget git git-gui ctags cscope libboost-dev vim
RUN wget -O /tmp/cuda_8.0.61_375.26_linux-run https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
RUN chmod +x /tmp/cuda_8.0.61_375.26_linux-run
RUN /tmp/cuda_8.0.61_375.26_linux-run -silent -toolkit
RUN git clone https://github.com/DebashisGanguly/gpgpu-sim_UVMSmart.git /root/gpgpu-sim_UVMSmart
RUN /bin/bash -c "cd /root/gpgpu-sim_UVMSmart; source setup_environment; make -j20; cd benchmarks/; ./setup_config.sh GeForceGTX1080Ti; make; find . -name 'run' | xargs chmod +x"

