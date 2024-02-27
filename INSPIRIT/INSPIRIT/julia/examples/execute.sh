#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2021       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

set -x
export JULIA_LOAD_PATH=/root/INSPIRIT/./julia/src:$JULIA_LOAD_PATH
export STARPU_BUILD_DIR=/root/INSPIRIT
export STARPU_SRC_DIR=/root/INSPIRIT/.
export STARPU_JULIA_LIB=/root/INSPIRIT/julia/src/.libs/libstarpujulia-1.3
export STARPU_JULIA_BUILD=/root/INSPIRIT/julia
export LD_LIBRARY_PATH=/root/INSPIRIT/julia/src/.libs/:$LD_LIBRARY_PATH
export JULIA_NUM_THREADS=8
export STARPU_NOPENCL=0
export STARPU_SCHED=dmda

srcdir=/root/INSPIRIT/./julia/examples

rm -f genc*.c gencuda*.cu genc*.o

if test "$1" == "-calllib"
then
    shift
    pwd
    rm -f extern_tasks.so
    make -f /root/INSPIRIT/julia/src/dynamic_compiler/Makefile extern_tasks.so SOURCES_CPU=$srcdir/$1
    shift
    export JULIA_TASK_LIB=$PWD/extern_tasks.so
fi

srcfile=$1
if test ! -f $srcdir/$srcfile
then
    echo "Error. File $srcdir/$srcfile not found"
    exit 1
fi
shift
#cd $srcdir/$(dirname $srcfile)
#exec  $(basename $srcfile) $*
exec  $srcdir/$srcfile $*

