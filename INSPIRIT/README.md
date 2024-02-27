# INSPIRIT

`INSPIRIT/`: StarPU integrated with INSPIRIT 
`INSPIRIT_scripts/`: Scripts used to collect performance data  

## Experiment Environment

| environment |                                                  |
| ----------- | ------------------------------------------------ |
| Platform    | X86                                              |
| CPU         | Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz        |
| Core        | 26                                               |
| CUDA 0      | NVIDIA GeForce RTX 3090                          |
| CUDA 1      | Tesla V100-PCIE-32GB                             |
| System      | Ubuntu 18.04.6 LTS Linux gpu2 4.15.0-101-generic |

## Compile StarPU

We compile StarPU through `./configure --disable-fxt --enable-blas-lib=mkl --with-mkl-ldflags="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" && make`.

Our core modifications are as follows.

`INSPIRIT/INSPIRIT/src/sched_policies/fifo_queues.c`: we define function `starpu_st_fifo_taskq_pop_first_ready_efi_task` to pop task with a high `prior_efi`.

`INSPIRIT/INSPIRIT/src/sched_policies/deque_modeling_policy_data_aware.c`: we modify function `_dmda_pop_task` , which contains our core scheduling algorithm.

## Get Results

1. Generate DAG by analyzing trace

We first compile starpu with fxt enabled through `./configure --enable-blas-lib=mkl --with-mkl-ldflags="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" && make`. Then we can generate DAGs under `1_gen_trace_dag/cholesky/dag` in defined format.
```bash
cd INSPIRIT/INSPIRIT_scripts/1_gen_trace_dag
./gen_trace_dag.sh {task_name} {task_path} {task_script} {NBLOCKS}
eg. ./gen_trace_dag.sh "cholesky" "/root/starpu_trace_view/examples/cholesky" "cholesky_implicit" 6
```

2. Generate priors by analyzing DAG

```bash
cd INSPIRIT/INSPIRIT_scripts/2_gen_prior_res
./gen_priors.sh {task_name} {task_path} {task_script} {NBLOCKS}
eg. ./gen_priors.sh "cholesky" "/root/starpu_trace_view/examples/cholesky" "cholesky_implicit" 6
```

3. Run baseline and SOTA

```bash
cd INSPIRIT/INSPIRIT_scripts/3_gen_dif_env
python gen_dif_env.py --N {execution times} --N_skip {warm up times} --NBLOCKS {task_scale} --task_dir {task_path} --task_script {task_script} --task_name {task_name} --res_name {result file name}
eg. python gen_dif_env.py --N 10 --N_skip 5 --NBLOCKS 64 --task_dir "/root/INSPIRIT/examples/cholesky" --task_script "cholesky_implicit" --task_name "cholesky" --res_name "dif_env_gflops.txt"
```

The format of the generated results are:

```
NBLOCKS 	NCPU 	NCUDA	TCUDA  	SCHED 	PRIOR	GFLOPS_5	GFLOPS_6	GFLOPS_7	GFLOPS_8	GFLOPS_9	GFLOPS
2       	26   	2    	0, 1   	dmda  	0    	87.1    	87.8    	80.7    	87.3    	87.5    	86.1  
...
# NBLOCKS for task scale (960*960*NBLOCKS)
# NCPU, NCUDA, TCUDA for CPU number, GPU number, GPU type (0 for 3090; 1 for V100) respectively
# SCHED, PRIOR for scheduling policies (dmda 0 for "dmda", dmdap 0 for "Base priority", dmdap 1 for "Kut polyhedral tool", dmdap 2 for "PaRSEC priority")
# GFLOPS_x for execution x's GFLOPS
# GFLOPS for the average GFLOPS of executions
```

4. Run INSPIRIT

```bash
cd INSPIRIT/INSPIRIT_scripts/4_auto_opt_dflops
python scaling.py --N {execution times} --N_skip {warm up times} --NBLOCKS {task_scale} --task_dir {task_path} --task_script {task_script} --task_name {task_name} --res_name {result file name}
eg. python scaling.py --N 4 --N_skip 3 --NBLOCKS 64 --task_dir "/root/INSPIRIT/examples/cholesky" --task_script "cholesky_implicit" --task_name "cholesky" --res_name "commands.log"
```
The format of the generated results are:

```
53760_26_2_0,1_dmdap_-1
current max gflops: 20926.4; p: 0

# Under the task scale of 53760 and a hardware environment consisting of 26 CPUs and 2 GPUs (3090 and V100), the best handcrafted priority policy, "base priority," achieves a GFLOPS value of 20926.4.

bash cholesky_implicit -size 53760 -nblocks 56 -priority_attribution_p -1 -priors "priors.txt" -priors_abi "priors_abi.txt" -priors_efi "priors_efi.txt" -nready_k_list [3] -nready_lb_list [10,10] -auto_opt 1
19807.0 19807.0 
bash cholesky_implicit -size 53760 -nblocks 56 -priority_attribution_p -1 -priors "priors.txt" -priors_abi "priors_abi.txt" -priors_efi "priors_efi.txt" -nready_k_list [3] -nready_lb_list [10,20] -auto_opt 1
20103.1 20103.1 
bash cholesky_implicit -size 53760 -nblocks 56 -priority_attribution_p -1 -priors "priors.txt" -priors_abi "priors_abi.txt" -priors_efi "priors_efi.txt" -nready_k_list [3] -nready_lb_list [10,50] -auto_opt 1
20889.1 20889.1 
bash cholesky_implicit -size 53760 -nblocks 56 -priority_attribution_p -1 -priors "priors.txt" -priors_abi "priors_abi.txt" -priors_efi "priors_efi.txt" -nready_k_list [3] -nready_lb_list [10,100] -auto_opt 1
20512.6 20512.6 
bash cholesky_implicit -size 53760 -nblocks 56 -priority_attribution_p -1 -priors "priors.txt" -priors_abi "priors_abi.txt" -priors_efi "priors_efi.txt" -nready_k_list [3] -nready_lb_list [10,200] -auto_opt 1
22192.2 22192.2 
[FOUND!!!] 22192.2

# We discovered a superior scheduling policy under INSPIRIT, yielding a GFLOPS performance of 22192.2.

...
```



