NVCC ?= nvcc
ARCH ?= sm_89
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++14 -Xcompiler -O3,-Wall

.PHONY: all clean

all: ed_replay ed_batchbench

ed_replay: ed_replay.cu ed_semi64.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ ed_replay.cu

ed_batchbench: ed_batchbench.cu ed_semi64.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ ed_batchbench.cu

clean:
	rm -f ed_replay ed_batchbench
