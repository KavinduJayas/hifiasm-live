NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_89
NVCCFLAGS = -O3 -arch=$(ARCH) -std=c++14 -Xcompiler -O3,-Wall

.PHONY: all clean

all: ed_replay

ed_replay: ed_replay.cu ed_semi64.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ ed_replay.cu

clean:
	rm -f ed_replay
