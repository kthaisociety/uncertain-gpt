# BNN: the CPU reference and the GPU port.
#
#   make            -> both binaries
#   make train_bnn      ./train_bnn        the C reference (checks + CPU training)
#   make train_bnn_gpu  ./train_bnn_gpu    the CUDA trainer
#
# The per-kernel development files live in dev/ and have their own Makefile.
# Nothing here depends on them; they are where each kernel was built and
# validated before being copied into train_bnn_gpu.cu.

NVCC       ?= nvcc
ARCH       ?= sm_75
NVCC_FLAGS  = -arch=$(ARCH) -O3 --use_fast_math
CFLAGS      = -O2 -Wall

all: train_bnn train_bnn_gpu

# ---- the CPU reference ------------------------------------------------------
# Deterministic, single-threaded, gradient-checked. This is the oracle: the
# thing every GPU kernel was validated against.
train_bnn: train_bnn.c
	gcc $(CFLAGS) $< -lm -o $@

# ---- the GPU trainers -------------------------------------------------------
# One binary per Bayesian optimizer. Everything they share — kernels, layout,
# forward, backward, dataloader — is in gpt2_kernels.cuh, so each .cu contains
# only its own optimizer and a fix to the network lands in all of them at once.
#
# Not linked against the CPU object: the kernels are validated in dev/, not here.
train_bnn_gpu: train_bnn_gpu.cu gpt2_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -DNBLOCKS=$(L) $< -lcublas -o $@

# Vadam (Khan et al. 2018).
train_bnn_vadam: train_bnn_vadam.cu gpt2_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -DNBLOCKS=$(L) $< -lcublas -o $@

# Deep ensemble baseline (Lakshminarayanan et al. 2017) — the control arm.
train_ensemble: train_ensemble.cu gpt2_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -DNBLOCKS=$(L) $< -lcublas -o $@

# Low-rank + diagonal covariance (Arm B). Rank is compile-time so the r-loop
# unrolls into registers:  make train_bnn_lowrank R=4   (default 8)
R ?= 8
L ?= 6
train_bnn_lowrank: train_bnn_lowrank.cu gpt2_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -DRANK=$(R) -DNBLOCKS=$(L) $< -lcublas -o $@

# ---- the scorecard ----------------------------------------------------------
# `make test` builds everything, runs every variant in both modes, and prints
# one comparison table. Exit code 0 only if every gate passes.
scorecard: scorecard.c
	gcc $(CFLAGS) $< -o $@

test: all train_bnn_vadam train_ensemble train_bnn_lowrank scorecard
	./scorecard

clean:
	rm -f train_bnn train_bnn_gpu train_bnn_vadam train_ensemble train_bnn_lowrank scorecard train_bnn_cpu.o

.PHONY: all clean test
