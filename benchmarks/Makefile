BENCHMARKS := $(shell find . -mindepth 2 -maxdepth 2 -type d)

default:
	for dir in $(BENCHMARKS) ; do make -C $$dir; done

clean:
	for dir in $(BENCHMARKS) ; do make clean -C $$dir; done
