#!/bin/bash
# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2021-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

benchmarkdir=/root/INSPIRIT/./starpupy/benchmark

modpath=/root/INSPIRIT/src/.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
pypath=/root/INSPIRIT/starpupy/src/build:$PYTHONPATH

LOADER=""
PYTHON=python3

if test -z "$LAUNCHER"
then
    LAUNCHER="mpiexec -np 2"
fi
mpi=""
gdb=""

read_arg()
{
    do_shift=0
    if test "$1" == "--valgrind"
    then
	export PYTHONMALLOC=malloc
	LOADER="valgrind --track-origins=yes"
	do_shift=1
    elif test "$1" == "--gdb"
    then
	gdb="gdb"
	if test "$mpi" == "mpi"
	then
           LOADER="$LAUNCHER xterm -sl 10000 -hold -e gdb --args"
	else
	    LOADER="gdb --args"
	fi
	do_shift=1
    elif test "$1" == "--mpirun"
    then
	mpi="mpi"
	if test "$gdb" == "gdb"
	then
           LOADER="$LAUNCHER xterm -sl 10000 -hold -e gdb --args"
	else
           LOADER="$LAUNCHER"
	fi
	do_shift=1
    fi
}

for x in $*
do
    read_arg $x
    if test $do_shift == 1
    then
	shift
    fi
done
for x in $LOADER_ARGS
do
    read_arg $x
done

benchmarkfile=$1
if test -f $benchmarkfile
then
    pythonscript=$benchmarkfile
elif test -f $benchmarkdir/$benchmarkfile
then
    pythonscript=$benchmarkdir/$benchmarkfile
else
    echo "Error. Python script $benchmarkfile not found in current directory or in $benchmarkdir"
    exit 1
fi
shift

set -x
PYTHONPATH=$pypath LD_LIBRARY_PATH=$modpath exec $LOADER $PYTHON $pythonscript $*

