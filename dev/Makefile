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

# The two Vadam kernels are the exception to the rule above: they live in
# ../train_bnn_vadam.cu, not in train_bnn.c, so these files include it and
# define UNCERTAIN_AS_LIBRARY — the guard the Python extension already uses to
# compile out an arm's main() — to test the REAL kernels, not a copy. (The
# -DSKIP_MAIN above is train_bnn.c's own, separate guard.) That pulls in
# gpt2_kernels.cuh, whose cuda_check_/ceil_div collide with common.h's, so
# common.h is deliberately not used and train_bnn_cpu.o is not linked.
#   make vadam && ./vadam_update && ./vadam_sample
VADAM = vadam_update vadam_sample
vadam: $(VADAM)
$(VADAM): %: %.cu ../train_bnn_vadam.cu ../gpt2_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -DNBLOCKS=6 $< -lcublas -o $@

# Finite-difference check on the L>1 block loop. Uses gpt2_kernels.cuh's
# forward/backward directly — no optimizer, no train_bnn.c — for the same
# common.h reason as the two files above.
block_loop: block_loop.cu ../gpt2_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) $< -lcublas -o $@

# Equivalence check for the fused parameter updates: the fused kernel against
# the kernels it replaced, on identical inputs. Includes TWO arms in one
# translation unit — the same thing uncertain/_C.cu does — so both fusions are
# checked by one binary.
#   make -C dev fused && ./dev/fused
#   make -C dev fused fused_nofma && ./dev/fused && ./dev/fused_nofma
# TWO builds on purpose. `fused` uses production flags and allows a few ULP,
# because nvcc contracts multiply-adds differently in the fused kernel's
# unrolled r-loop than in a standalone AdamW. `fused_nofma` removes that
# licence with -fmad=false and requires EXACT bit equality — which is what
# turns "the difference is only contraction" from a claim into a result.
FUSED_SRC = fused.cu ../train_bnn_gpu.cu ../train_bnn_vadam.cu \
            ../train_bnn_lowrank.cu ../gpt2_kernels.cuh
fused: $(FUSED_SRC)
	$(NVCC) $(NVCC_FLAGS) -DNBLOCKS=6 -DRANK=8 $< -lcublas -o $@

fused_nofma: $(FUSED_SRC)
	$(NVCC) $(NVCC_FLAGS) -fmad=false -DFUSED_NO_FMAD -DNBLOCKS=6 -DRANK=8 \
	    $< -lcublas -o $@

clean:
	rm -f $(KERNELS) $(VADAM) block_loop fused fused_nofma train_bnn_cpu.o
.PHONY: all clean vadam
