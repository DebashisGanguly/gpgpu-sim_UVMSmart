#!/usr/bin/make

# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -g -Wno-deprecated-gpu-targets 

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(NVCC) $(NVCC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -lgomp -lcudart -lm

%.o: %.[c]
	$(CC) $(CC_FLAGS) $< -c

kmeans_cuda.o: kmeans_cuda.cu 
	$(NVCC) $(NVCC_FLAGS) -c kmeans_cuda.cu

clean:
	rm -f *.o *~ kmeans *.txt gpgpusim_power_report* _cuobjdump_*
