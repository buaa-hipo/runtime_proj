# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

AM_CFLAGS = $(GLOBAL_AM_CFLAGS)
AM_CXXFLAGS = $(GLOBAL_AM_CXXFLAGS)
AM_FFLAGS = $(GLOBAL_AM_FFLAGS)
AM_FCFLAGS = $(GLOBAL_AM_FCFLAGS)

recheck:
	-cat /dev/null

showfailed:
	@-cat /dev/null

showcheck:
	-cat /dev/null

showsuite:
	-cat /dev/null
