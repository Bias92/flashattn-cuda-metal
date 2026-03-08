from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attn_cuda",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_cuda",
            sources=["cuda/flash_attn_kernel.cu"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",  # RTX 4060 Ti (Ada Lovelace)
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
