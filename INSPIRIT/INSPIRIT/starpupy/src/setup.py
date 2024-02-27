# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2022  Universit'e de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
from setuptools import setup, Extension
import sys

numpy_dir = ''
if numpy_dir != '':
    numpy_include_dir = [numpy_dir]
else:
    numpy_include_dir = []

cppflags = " -I/usr/local/cuda/include  -I/usr/local/cuda/include -DSTARPU_OPENCL_DATADIR=${prefix}/share/starpu/opencl -DCL_USE_DEPRECATED_OPENCL_1_1_APIS    "
compile_args = cppflags.split(' ')
extra_compile_args = []
for f in compile_args:
    if f:
        extra_compile_args.append(f)

ver = sys.version_info
libpython = 'python%s.%s%s' % (ver.major, ver.minor, sys.abiflags)

starpupy = Extension('starpu.starpupy',
                     include_dirs = ['/root/INSPIRIT/./include', '/root/INSPIRIT/include', '/root/INSPIRIT/./starpupy/src'] + numpy_include_dir,
                     libraries = ['starpu-1.3', libpython],
                     extra_compile_args = extra_compile_args,
                     extra_link_args = ['-shared'],
                     library_dirs = ['/root/INSPIRIT/src/.libs'],
	             sources = ['starpu/starpu_task_wrapper.c', 'starpu/starpupy_handle.c', 'starpu/starpupy_interface.c', 'starpu/starpupy_buffer_interface.c', 'starpu/starpupy_numpy_filters.c'])

setup(
    name = 'starpupy',
    version = '0.5',
    description = 'Python bindings for StarPU',
    author = 'StarPU team',
    author_email = 'starpu-devel@inria.fr',
    url = 'https://starpu.gitlabpages.inria.fr/',
    license = 'GPL',
    platforms = 'posix',
    ext_modules = [starpupy],
    packages = ['starpu'],
    )
