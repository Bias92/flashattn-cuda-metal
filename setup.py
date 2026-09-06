from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash-attn-cuda",
    packages=[],
    ext_modules=[
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
