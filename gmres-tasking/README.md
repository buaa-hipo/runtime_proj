# OpenMP Taskified GMRES

## Experiment Environment

| environment |                                                  |
| ----------- | ------------------------------------------------ |
| Platform    | X86                                              |
| CPU         | Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz        |
| Core        | 26                                               |
| CUDA 0      | NVIDIA GeForce RTX 3090                          |
| CUDA 1      | Tesla V100-PCIE-32GB                             |
| System      | Ubuntu 18.04.6 LTS Linux gpu2 4.15.0-101-generic |

## Prerequisites

- Intel MKL
- Clang++

Following instructions are executed with MKL environments set.

## Compilation

```bash
clang++ -o omp_gmres ./omp_gmres.cpp ./mmio.c -fopenmp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -O3 -g
```

## Execution

```bash
OMP_NUM_THREADS=4 ./omp_gmres
```

- To change target matrix, modifiy the `filename` varible in the main function of source code `omp_gmres.cpp`, the filename should point to a matrix market format matrix