# make            build everything this machine can build
# make cpp        just the plain C++ programs (needs g++)
# make cuda       just the CUDA programs (needs nvcc, running them needs an NVIDIA GPU)
# make clean      delete build/
#
# Every folder like 01_cpp_basics/ or 02_vector_add/ is picked up automatically:
# a .cpp file becomes build/<folder>/<name>, a .cu file the same.
CXX  ?= g++
NVCC ?= nvcc
ARCH ?= sm_75   # T4. Use sm_86 for an RTX 30xx, sm_89 for an RTX 40xx

CPP_SRCS := $(wildcard */*.cpp)
CU_SRCS  := $(wildcard */*.cu)
CPP_BINS := $(patsubst %.cpp,build/%,$(CPP_SRCS))
CU_BINS  := $(patsubst %.cu,build/%,$(CU_SRCS))

.PHONY: all cpp cuda clean
all: cpp cuda
cpp: $(CPP_BINS)
cuda: $(CU_BINS)

build/%: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wextra $< -o $@

build/%: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) -std=c++17 -O2 -arch=$(ARCH) $< -o $@

clean:
	rm -rf build
