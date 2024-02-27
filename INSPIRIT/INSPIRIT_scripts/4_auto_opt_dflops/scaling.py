import os
import subprocess
from datetime import datetime
import shutil
import math
import argparse
cur_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=4, required=False)
parser.add_argument('--N_skip', type=int, default=3, required=False) # warmup
parser.add_argument('--task_dir', type=str, default="/root/INSPIRIT/examples/cholesky", required=False)
parser.add_argument('--task_script', type=str, default="cholesky_implicit", required=False)
parser.add_argument('--task_name', type=str, default="cholesky", required=False)
parser.add_argument('--res_name', type=str, default="commands.log", required=False)
parser.add_argument('--NBLOCKS', type=int, default=64, required=False)
args = parser.parse_args()

NBLOCKS = args.NBLOCKS
scale_size = [int(2 * (i + 1)) for i in range(NBLOCKS//2)]


N = args.N # 设置多次实验次数
N_skip = args.N_skip # 设置跳过的实验次数
task_dir = args.task_dir
task_script = args.task_script      
task_name = args.task_name                                
res_dir = f"{cur_path}/{task_name}"
log_dir = f"{res_dir}/{args.res_name}"

hardware_env = [
    (NCPU, NCUDA, TCUDA)
    for (NCPU, NCUDA, TCUDA) in [(0, 2, "0,1"), (26, 2, "0,1"), (26, 1, "1"), (26, 1, "0")]
    
]
scheduling_params = [
    (SCHED, PRIOR)
    for (SCHED, PRIOR) in [("dmdap", -1)]
]
nready_k_list = []
nready_lb_list = []
k_list = [3, 5]
lb_list = [10, 20, 50, 100, 200, 300, 400, 500]
for k in k_list:
    for lb1 in lb_list:
        for lb2 in lb_list:
            tmp_lb_list = [lb1, lb2]
            nready_k_list.append([k])
            nready_lb_list.append(tmp_lb_list)
nready_k_list_max = []
nready_lb_list_max = []
# (nready_k_list_max, nready_lb_list_max)
opt_args = []
# cmd = f"bash {task_script} -size {nblocks * 960} -nblocks {nblocks} -priority_attribution_p {PRIOR} -priors \"priors.txt\" -priors_abi \"priors_abi.txt\" -priors_efi \"priors_efi.txt\" 
# -nready_k_list {'['+','.join(map(str, nready_k_list_max))+']'} -nready_lb_list {'['+','.join(map(str, nready_lb_list_max))+']'} 
# -nready_ub_list {'['+','.join(map(str, nready_ub_list_max))+']'} -nready_pop_list {'['+','.join(map(str, nready_pop_list_max))+']'} 
# -nready_alpha_list {'['+','.join(map(str, nready_alpha_list_max))+']'} -nready_beta_list {'['+','.join(map(str, nready_beta_list_max))+']'} -auto_opt {auto_opt}"
for ii in range(len(nready_lb_list)):
    nready_lb_list_max = nready_lb_list[ii]
    nready_k_list_max = nready_k_list[ii]
    opt_args.append((nready_k_list_max, nready_lb_list_max))

cond2perf = {}
for NBLOCKS in scale_size:
    cond2perf[NBLOCKS] = {}
    for HARDWARE in hardware_env:
        cond2perf[NBLOCKS][HARDWARE] = {}
        # baseline_best_gflops
        cond2perf[NBLOCKS][HARDWARE]["baseline_best_gflops"] = 0
        cond2perf[NBLOCKS][HARDWARE]["baseline_best_p"] = 0
        for SCHEDULE in scheduling_params:
            cond2perf[NBLOCKS][HARDWARE][SCHEDULE] = 0
            # dmda_gflops, dmdap_0_gflops, dmdap_1_gflops, dmdap_2_gflops
            cond2perf[NBLOCKS][HARDWARE][SCHEDULE] = 0
            
def init_cond2perf():
    import pandas as pd

    # read from txt
    txt_path_1 = f'{cur_path}/../3_gen_dif_env/{task_name}/dif_env_gflops.txt'
    # txt_path_2 = f'{cur_path}/../3_gen_dif_env/{task_name}/dif_env_gflops_2.txt'

    df1 = pd.read_csv(txt_path_1, sep='\t', header=None, skiprows=1, engine='python')
    
    df = df1
    
    # df2 = pd.read_csv(txt_path_2, sep='\t', header=None, skiprows=1, engine='python')
    # merged_df = pd.concat([df1, df2], ignore_index=True)
    # xlsx_path = f"{cur_path}/{task_name}/graph2_unit_1.xlsx"

    # merged_df.to_excel(xlsx_path, index=False)
    # df = pd.read_excel(xlsx_path)

    result = []
    temp = []
    num_columns = df.shape[1]
    for index, row in df.iterrows():
        for i in range(num_columns):
            temp.append(row[i])
        result.append(temp)
        temp = []
    result.sort(key=lambda x: (x[0], x[1], -x[2]))

    # init key
    # scale_size = [
    #     2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
    #     22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
    #     42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
    #     62, 64
    # ]
    hardware_env = [
        (NCPU, NCUDA, TCUDA)
        for (NCPU, NCUDA, TCUDA) in [(0, 2, "0,1"), (26, 2, "0,1"), (26, 1, "0"), (26, 1, "1")]
    ]
    scheduling_params = [
        (SCHED, PRIOR)
        for (SCHED, PRIOR) in [("dmda", 0), ("dmdap", -1), ("dmdap", 0), ("dmdap", 1), ("dmdap", 2)]
    ]

    # fill the dict
    def sort_second_small(a: list) -> float:
        selected_elements = a[6:-1]
        sorted_elements = sorted(selected_elements)
        if len(sorted_elements) == 1:
            return float(sorted_elements[0])
        return float(sorted_elements[1])

    # print(sort_second_small(result[0]))

    def sort_one_big_value(a, b, c, d, e) -> float:
        selected_elements = [a, b, c, d, e]
        sorted_elements = sorted(selected_elements)
        return float(sorted_elements[4])

    def sort_one_big_index(a, b, c, d, e) -> int:
        selected_elements = [a, b, c, d, e]
        sorted_elements = sorted(selected_elements)
        return selected_elements.index(sorted_elements[4]) - 2

    cond2perf = {}
    i = 0
    for NBLOCKS in scale_size:
        cond2perf[NBLOCKS] = {}
        for HARDWARE in hardware_env:
            cond2perf[NBLOCKS][HARDWARE] = {}
            for SCHEDULE in scheduling_params:
                if SCHEDULE == ("dmdap", -1):
                    cond2perf[NBLOCKS][HARDWARE][SCHEDULE] = 0
                else:
                    cond2perf[NBLOCKS][HARDWARE][SCHEDULE] = sort_second_small(result[i])
                    i = i + 1
            # baseline_best_gflops
            cond2perf[NBLOCKS][HARDWARE]["baseline_best_gflops"] = sort_one_big_value(
                cond2perf[NBLOCKS][HARDWARE][("dmda", 0)],
                cond2perf[NBLOCKS][HARDWARE][("dmdap", -1)],
                cond2perf[NBLOCKS][HARDWARE][("dmdap", 0)],
                cond2perf[NBLOCKS][HARDWARE][("dmdap", 1)],
                cond2perf[NBLOCKS][HARDWARE][("dmdap", 2)])

            cond2perf[NBLOCKS][HARDWARE]["baseline_best_p"] = sort_one_big_index(
                cond2perf[NBLOCKS][HARDWARE][("dmda", 0)],
                cond2perf[NBLOCKS][HARDWARE][("dmdap", -1)],
                cond2perf[NBLOCKS][HARDWARE][("dmdap", 0)],
                cond2perf[NBLOCKS][HARDWARE][("dmdap", 1)],
                cond2perf[NBLOCKS][HARDWARE][("dmdap", 2)])
    return cond2perf

def get_curconf_gflops(NBLOCKS, HARDWARE, SCHEDULE, N, Nskip, OPTARG):
    # 切换到 task_dir 目录
    os.chdir(task_dir)
    
    nblocks = NBLOCKS
    (NCPU, NCUDA, TCUDA) = HARDWARE
    (SCHED, PRIOR) = SCHEDULE
    (nready_k_list_max, nready_lb_list_max) = OPTARG
    
    os.environ["STARPU_NCPU"] = str(NCPU)
    os.environ["STARPU_NCUDA"] = str(NCUDA)
    os.environ["CUDA_VISIBLE_DEVICES"] = TCUDA
    os.environ["STARPU_SCHED"] = SCHED
    os.environ["STARPU_HOSTNAME"] = f"{NBLOCKS}_{NCPU}_{NCUDA}_{TCUDA}_{SCHED}_{PRIOR}"
    shutil.copy(f"{cur_path}/../2_gen_prior_res/cholesky/priors_abi/{NBLOCKS * 960}.txt", f"{task_dir}/priors.txt")
    shutil.copy(f"{cur_path}/../2_gen_prior_res/cholesky/priors_abi/{NBLOCKS * 960}.txt", f"{task_dir}/priors_abi.txt")
    shutil.copy(f"{cur_path}/../2_gen_prior_res/cholesky/priors_efi/{NBLOCKS * 960}.txt", f"{task_dir}/priors_efi.txt")
    os.system(f"rm /root/.starpu/sampling/codelets/45/*")
    os.system(f"rm /root/.starpu/sampling/bus/*")
    
    # 初始化结果列表
    results1 = []
    for i in range(N):
        if PRIOR == -1:
            auto_opt = 1
        else:
            auto_opt = 0
        cmd1 = f"bash {task_script} -size {nblocks * 960} -nblocks {nblocks} -priority_attribution_p {PRIOR} -priors \"priors.txt\" -priors_abi \"priors_abi.txt\" -priors_efi \"priors_efi.txt\" -nready_k_list {'['+','.join(map(str, nready_k_list_max))+']'} -nready_lb_list {'['+','.join(map(str, nready_lb_list_max))+']'} -auto_opt {auto_opt}"
        if (i == N-1):
            with open(log_dir, "a") as log_file:
                log_file.write(cmd1 + "\n") 
        output1 = subprocess.check_output(cmd1, shell=True, text=True)
        readGFlops = 0
        for line in output1.splitlines():
            if readGFlops == 1:
                # 提取 GFLOPS 数据
                if (len(line.split()) < 1):
                    gflops1 = 0.0
                else:
                    gflops1 = float(line.split()[-1])
                break
            if "GFlop/s" in line:
                readGFlops = 1
            print(line, end='')  # 显示输出到终端
        # 将结果添加到列表
        results1.append(gflops1)
    # 保存结果到文件
    output1_name = f"{nblocks * 960}_{NCPU}_{NCUDA}_{TCUDA}_{SCHED}_{PRIOR}_{'['+','.join(map(str, nready_k_list_max))+']'}_{','.join(map(str, nready_lb_list_max))}"
    newdir_path = os.path.join(res_dir, f"{nblocks * 960}")
    os.system(f"mkdir -p {newdir_path}")
    output1_path = os.path.join(res_dir, f"{nblocks * 960}", output1_name+".txt")
    with open(output1_path, "w") as f:
        f.write(output1)
    # 丢弃第一次实验的结果
    results1 = results1[Nskip:]
    # 计算 N-1 次结果的均值
    if len(results1) == N - Nskip:
        avg_gflops1 = sum(results1) / (N - Nskip)
    else:
        avg_gflops1 = 0.0
    with open(log_dir, "a") as log_file:
        log_file.write(str(avg_gflops1) + " ") 
        for i in range(len(results1)):
            log_file.write(str(results1[i]) + " ")
        log_file.write("\n")
    
    os.system(f"rm /root/.starpu/sampling/codelets/45/*")
    os.system(f"rm /root/.starpu/sampling/bus/*")
    
    return avg_gflops1

if __name__ == "__main__":
    cond2perf = init_cond2perf()
    
    
    for NBLOCKS in scale_size:
        for HARDWARE in hardware_env:
            for SCHEDULE in scheduling_params:
                
                (NCPU, NCUDA, TCUDA) = HARDWARE
                (SCHED, PRIOR) = SCHEDULE
                # breakpoint()
                baseline_best_gflops = cond2perf[NBLOCKS][HARDWARE]["baseline_best_gflops"]
                baseline_best_p = cond2perf[NBLOCKS][HARDWARE]["baseline_best_p"] 
                
                with open(log_dir, "a") as log_file:
                    log_file.write(f"{NBLOCKS * 960}_{NCPU}_{NCUDA}_{TCUDA}_{SCHED}_{PRIOR}" + "\n")
                    log_file.write(f"current max gflops: {baseline_best_gflops}; p: {baseline_best_p}\n") 
                
                for OPTARG in opt_args:
                    # generate gflops
                    avg_gflops1 = get_curconf_gflops(NBLOCKS, HARDWARE, SCHEDULE, N, N_skip, OPTARG)
                    if avg_gflops1 > baseline_best_gflops:
                        with open(log_dir, "a") as log_file:
                            log_file.write(f"[FOUND!!!] {avg_gflops1}\n") 
                        break
                
                    