"""Build uncertain._C from the repository's existing CUDA sources.

Compiled on install; nvcc is required. That is a deliberate trade — a fast
artifact for a few serious users rather than a portable slow one for many —
so there is no pure-PyTorch fallback path and there never should be.

    pip install -e . --no-build-isolation

`--no-build-isolation` because this file imports the torch that is already
installed. With isolation, pip would build in a clean environment, find no
torch, and (if torch were declared a build requirement) download a second copy
of it that the extension would then be linked against but not run with.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))

# sm_75 by default, matching the Makefile's ARCH. Override with either:
#     UNCERTAIN_CUDA_ARCH=8.6 pip install -e . --no-build-isolation
#     TORCH_CUDA_ARCH_LIST="7.5;8.6" pip install -e . --no-build-isolation
#
# Setting this also stops torch probing the local GPU to guess arch flags,
# which is what lets the package build on a machine with no usable card.
_arch = os.environ.get("UNCERTAIN_CUDA_ARCH")
if _arch:
    os.environ["TORCH_CUDA_ARCH_LIST"] = _arch
else:
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

# Escape hatch for host-compiler mismatches, e.g. CUDA rejecting a newer gcc:
#     UNCERTAIN_NVCC_EXTRA_FLAGS="-allow-unsupported-compiler" pip install -e . \
#         --no-build-isolation
# Deliberately opt-in. Silently allowing an unsupported host compiler is how you
# get a build that succeeds and then miscompiles.
_extra_nvcc = os.environ.get("UNCERTAIN_NVCC_EXTRA_FLAGS", "").split()

setup(
    ext_modules=[
        CUDAExtension(
            name="uncertain._C",
            # One translation unit. It #includes the four arm .cu files, which
            # in turn include gpt2_kernels.cuh — compiling them separately
            # would give four definitions of every shared __global__ kernel.
            sources=["uncertain/_C.cu"],
            include_dirs=[HERE],
            libraries=["cublas"],
            extra_compile_args={
                # The Makefile's flags. --use_fast_math is not cosmetic: the
                # kernels were validated against the CPU reference with it, and
                # the low-rank arm's bitwise checks depend on host and device
                # agreeing on which transcendentals are in play.
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"] + _extra_nvcc,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
