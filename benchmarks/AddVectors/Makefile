#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets add_vectors.cu -L /usr/local/cuda/lib -lcudart -o add_vectors
	nvcc -g -Wno-deprecated-gpu-targets add_vectors_old.cu -L /usr/local/cuda/lib -lcudart -o add_vectors_old
clean:
	rm -rf add_vectors add_vectors_old
	rm -rf gpgpusim_power_report*
	rm -rf _cuobjdump_*
	rm -rf *.txt
