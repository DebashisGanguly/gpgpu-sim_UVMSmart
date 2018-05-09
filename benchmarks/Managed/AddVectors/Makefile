#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets add_vectors.cu -L /usr/local/cuda/lib -lcudart -o add_vectors
clean:
	rm -rf add_vectors
	rm -rf gpgpusim_power_report*
	rm -rf _cuobjdump_*
	rm -rf *.txt
