from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='flash_attn',
    ext_modules=[
        CUDAExtension(
            name='flash_attn',
            sources=[
                'csrc/flash_attn_v1_naive.cu',
                'csrc/flash_attn_v1_smem.cu',
                'csrc/flash_attn_v1_warp.cu',
                'csrc/bindings.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-arch=sm_86', '-O3', '--use_fast_math', '-std=c++17'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
