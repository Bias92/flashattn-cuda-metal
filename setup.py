from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash-attn-cuda",
    ext_modules=[
        # Baseline FP32 kernel
        CUDAExtension(
            name="flash_attn_cuda",
            sources=["cuda/flash_attn_kernel.cu"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
        # WMMA Tensor Core optimized kernel
        CUDAExtension(
            name="flash_attn_wmma",
            sources=["cuda/flash_attn_wmma.cu"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
        # mma.sync + ldmatrix, register-resident softmax
        CUDAExtension(
            name="flash_attn_mma",
            sources=["cuda/flash_attn_mma.cu"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
        # mma.sync kernel + cp.async K/V double buffering
        CUDAExtension(
            name="flash_attn_mma_db",
            sources=["cuda/flash_attn_mma_db.cu"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
        # db + address strength reduction
        CUDAExtension(
            name="flash_attn_mma_db_addr",
            sources=["cuda/flash_attn_mma_db_addr.cu"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
        # Custom CUDA kernel used by bench/compare_pytorch.py
        CUDAExtension(
            name="attention_forward_cuda",
            sources=["cuda/attention_forward.cu"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
