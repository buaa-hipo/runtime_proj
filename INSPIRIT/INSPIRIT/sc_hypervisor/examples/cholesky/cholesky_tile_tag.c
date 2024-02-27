/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "cholesky.h"

struct starpu_perfmodel chol_model_potrf;
struct starpu_perfmodel chol_model_trsm;
struct starpu_perfmodel chol_model_syrk;
struct starpu_perfmodel chol_model_gemm;

/* A [ y ] [ x ] */
float *A[NMAXBLOCKS][NMAXBLOCKS];
starpu_data_handle_t A_state[NMAXBLOCKS][NMAXBLOCKS];

/*
 *	Some useful functions
 */

static struct starpu_task *create_task(starpu_tag_t id)
{
	struct starpu_task *task = starpu_task_create();
		task->cl_arg = NULL;
		task->use_tag = 1;
		task->tag_id = id;

	return task;
}

/*
 *	Create the codelets
 */

static struct starpu_codelet cl_potrf =
{
	.modes = { STARPU_RW },
	.cpu_funcs = {chol_cpu_codelet_update_potrf},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_potrf},
#endif
	.nbuffers = 1,
	.model = &chol_model_potrf
};

static struct starpu_task * create_task_potrf(unsigned k, unsigned nblocks)
{
/*	FPRINTF(stdout, "task 11 k = %d TAG = %llx\n", k, (TAG_POTRF(k))); */

	struct starpu_task *task = create_task(TAG_POTRF(k));
	
	task->cl = &cl_potrf;

	/* which sub-data is manipulated ? */
	task->handles[0] = A_state[k][k];

	/* this is an important task */
	task->priority = STARPU_MAX_PRIO;

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_POTRF(k), 1, TAG_GEMM(k-1, k, k));
	}

	return task;
}

static struct starpu_codelet cl_trsm =
{
	.modes = { STARPU_R, STARPU_RW },
	.cpu_funcs = {chol_cpu_codelet_update_trsm},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_trsm},
#endif
	.nbuffers = 2,
	.model = &chol_model_trsm
};

static void create_task_trsm(unsigned k, unsigned j)
{
	int ret;

	struct starpu_task *task = create_task(TAG_TRSM(k, j));

	task->cl = &cl_trsm;	

	/* which sub-data is manipulated ? */
	task->handles[0] = A_state[k][k];
	task->handles[1] = A_state[j][k];

	if (j == k+1)
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_TRSM(k, j), 2, TAG_POTRF(k), TAG_GEMM(k-1, k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG_TRSM(k, j), 1, TAG_POTRF(k));
	}

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

static struct starpu_codelet cl_syrk =
{
	.modes = { STARPU_R, STARPU_RW },
	.cpu_funcs = {chol_cpu_codelet_update_syrk},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_syrk},
#endif
	.nbuffers = 2,
	.model = &chol_model_syrk
};

static struct starpu_codelet cl_gemm =
{
	.modes = { STARPU_R, STARPU_R, STARPU_RW },
	.cpu_funcs = {chol_cpu_codelet_update_gemm},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {chol_cublas_codelet_update_gemm},
#endif
	.nbuffers = 3,
	.model = &chol_model_gemm
};

static void create_task_gemm(unsigned k, unsigned i, unsigned j)
{
	int ret;

/*	FPRINTF(stdout, "task 22 k,i,j = %d,%d,%d TAG = %llx\n", k,i,j, TAG_GEMM(k,i,j)); */

	struct starpu_task *task = create_task(TAG_GEMM(k, i, j));

	if (m == n)
	{
		task->cl = &cl_syrk;

		/* which sub-data is manipulated ? */
		task->handles[0] = A_state[i][k];
		task->handles[1] = A_state[j][i];
	}
	else
	{
		task->cl = &cl_gemm;

		/* which sub-data is manipulated ? */
		task->handles[0] = A_state[i][k];
		task->handles[1] = A_state[j][k];
		task->handles[2] = A_state[j][i];
	}

	if ( (i == k + 1) && (j == k +1) )
	{
		task->priority = STARPU_MAX_PRIO;
	}

	/* enforce dependencies ... */
	if (k > 0)
	{
		starpu_tag_declare_deps(TAG_GEMM(k, i, j), 3, TAG_GEMM(k-1, i, j), TAG_TRSM(k, i), TAG_TRSM(k, j));
	}
	else
	{
		starpu_tag_declare_deps(TAG_GEMM(k, i, j), 2, TAG_TRSM(k, i), TAG_TRSM(k, j));
	}

	ret = starpu_task_submit(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}



/*
 *	code to bootstrap the factorization 
 *	and construct the DAG
 */

static void cholesky_no_stride(void)
{
	int ret;

	struct timeval start;
	struct timeval end;

	struct starpu_task *entry_task = NULL;

	/* create all the DAG nodes */
	unsigned i,j,k;

	for (k = 0; k < nblocks; k++)
	{
		struct starpu_task *task = create_task_potrf(k, nblocks);
		/* we defer the launch of the first task */
		if (k == 0)
		{
			entry_task = task;
		}
		else
		{
			ret = starpu_task_submit(task);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
		
		for (j = k+1; j<nblocks; j++)
		{
			create_task_trsm(k, j);

			for (i = k+1; i<nblocks; i++)
			{
				if (i <= j)
					create_task_gemm(k, i, j);
			}
		}
	}

	/* schedule the codelet */
	gettimeofday(&start, NULL);
	ret = starpu_task_submit(entry_task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* stall the application until the end of computations */
	starpu_tag_wait(TAG_POTRF(nblocks-1));

	gettimeofday(&end, NULL);

	double timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	FPRINTF(stderr, "Computation took (in ms)\n");
	FPRINTF(stdout, "%2.2f\n", timing/1000);

	double flop = (1.0f*size*size*size)/3.0f;
	FPRINTF(stderr, "Synthetic GFlops : %2.2f\n", (flop/timing/1000.0f));
}

int main(int argc, char **argv)
{
	unsigned x, y;
	unsigned i, j;
	int ret;

	parse_args(argc, argv);
	assert(nblocks <= NMAXBLOCKS);

	FPRINTF(stderr, "BLOCK SIZE = %d\n", size / nblocks);

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_CUDA
	initialize_chol_model(&chol_model_potrf,"chol_model_potrf",cpu_chol_task_potrf_cost,cuda_chol_task_potrf_cost);
	initialize_chol_model(&chol_model_trsm,"chol_model_trsm",cpu_chol_task_trsm_cost,cuda_chol_task_trsm_cost);
	initialize_chol_model(&chol_model_syrk,"chol_model_syrk",cpu_chol_task_syrk_cost,cuda_chol_task_syrk_cost);
	initialize_chol_model(&chol_model_gemm,"chol_model_gemm",cpu_chol_task_gemm_cost,cuda_chol_task_gemm_cost);
#else
	initialize_chol_model(&chol_model_potrf,"chol_model_potrf",cpu_chol_task_potrf_cost,NULL);
	initialize_chol_model(&chol_model_trsm,"chol_model_trsm",cpu_chol_task_trsm_cost,NULL);
	initialize_chol_model(&chol_model_syrk,"chol_model_syrk",cpu_chol_task_syrk_cost,NULL);
	initialize_chol_model(&chol_model_gemm,"chol_model_gemm",cpu_chol_task_gemm_cost,NULL);
#endif

	/* Disable sequential consistency */
	starpu_data_set_default_sequential_consistency_flag(0);

	starpu_cublas_init();

	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y)
		{
#ifdef STARPU_HAVE_POSIX_MEMALIGN
			posix_memalign((void **)&A[y][x], 128, BLOCKSIZE*BLOCKSIZE*sizeof(float));
#else
			A[y][x] = malloc(BLOCKSIZE*BLOCKSIZE*sizeof(float));
#endif
			assert(A[y][x]);
		}
	}

	/* create a simple definite positive symetric matrix example
	 *
	 *	Hilbert matrix : h(i,j) = 1/(i+j+1) ( + n In to make is stable ) 
	 * */
	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	if (x <= y)
	{
		for (i = 0; i < BLOCKSIZE; i++)
		for (j = 0; j < BLOCKSIZE; j++)
		{
			A[y][x][i*BLOCKSIZE + j] =
				(float)(1.0f/((float) (1.0+(x*BLOCKSIZE+i)+(y*BLOCKSIZE+j))));

			/* make it a little more numerically stable ... ;) */
			if ((x == y) && (i == j))
				A[y][x][i*BLOCKSIZE + j] += (float)(2*size);
		}
	}

	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y)
		{
			starpu_matrix_data_register(&A_state[y][x], STARPU_MAIN_RAM, (uintptr_t)A[y][x], 
				BLOCKSIZE, BLOCKSIZE, BLOCKSIZE, sizeof(float));
		}
	}

	cholesky_no_stride();

	for (y = 0; y < nblocks; y++)
	for (x = 0; x < nblocks; x++)
	{
		if (x <= y)
		{
			starpu_data_unregister(A_state[y][x]);
			free(A[y][x]);
		}
	}

	starpu_cublas_shutdown();

	starpu_shutdown();
	return 0;
}


