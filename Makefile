# Makefile for matmul harness
# Tested on RunPod CUDA 12.x images

NVCC      = nvcc
SM        ?= 80          # default: A100 (sm_80). Override: make SM=89 for L40S, SM=86 for A6000
CFLAGS    = -O3 -arch=sm_$(SM) -lcublas -lcudart
TARGET    = matmul_harness

.PHONY: all clean run

all: $(TARGET)

$(TARGET): matmul_harness.cu
	$(NVCC) $(CFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

