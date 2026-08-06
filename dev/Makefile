# Per-kernel development files, llm.c dev/cuda style.
#   make residual_forward && ./residual_forward 1
#
# Each target links ../train_bnn.c (compiled with -DSKIP_MAIN, so no main) so
# every kernel is checked against the REAL CPU op, not a copy of it.
NVCC ?= nvcc
ARCH ?= sm_75
NVCC_FLAGS = -arch=$(ARCH) -O3 --use_fast_math -I.

KERNELS = residual_forward gelu_forward encoder_forward \
          bayes_materialize bayes_backward kl_backward adamw \
          residual_backward gelu_backward encoder_backward \
          layernorm_forward layernorm_backward softmax_crossentropy \
          crossentropy_softmax_backward matmul_forward matmul_backward attention_forward attention_backward

all: $(KERNELS)

train_bnn_cpu.o: ../train_bnn.c
	gcc -O2 -Wall -DSKIP_MAIN -c $< -o $@

%: %.cu common.h train_bnn_cpu.o
	$(NVCC) $(NVCC_FLAGS) $< train_bnn_cpu.o -lm -lcublas -o $@

clean:
	rm -f $(KERNELS) train_bnn_cpu.o
.PHONY: all clean
