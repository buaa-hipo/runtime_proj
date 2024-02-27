import os
import subprocess
import argparse
from datetime import datetime
cur_path = os.getcwd()

# 获取当前日期并格式化为字符串
date = datetime.now().strftime("%Y%m%d")

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=10, required=False)
parser.add_argument('--N_skip', type=int, default=5, required=False) # warmup
parser.add_argument('--task_dir', type=str, default="/root/INSPIRIT/examples/cholesky", required=False)
parser.add_argument('--task_script', type=str, default="cholesky_implicit", required=False)
parser.add_argument('--task_name', type=str, default="cholesky", required=False)
parser.add_argument('--res_name', type=str, default="dif_env_gflops.txt", required=False)
parser.add_argument('--NBLOCKS', type=int, default=64, required=False)

args = parser.parse_args()

# 定义实验参数
NBLOCKS = args.NBLOCKS
N = args.N
N_skip = args.N_skip
task_dir = args.task_dir
task_script = args.task_script                                     
res_dir = f"{cur_path}/{args.task_name}"
res_name = args.res_name

# 切换到 task_dir 目录
os.chdir(task_dir)

# 定义硬件环境参数和调度策略参数
hardware_env = [
    (NCPU, NCUDA, TCUDA)
    for (NCPU, NCUDA, TCUDA) in [(0, 2, "0, 1"), (26, 1, "1"), (26, 2, "0, 1"), (26, 1, "0")]
    # G1 纯GPU
    # for (NCPU, NCUDA, TCUDA) in [(0, 1, "0"), (0, 1, "1"), (0, 2, "0,1")]
    # G2 CPU+GPU
    # for (NCPU, NCUDA, TCUDA) in [(8, 1, "1"), (8, 1, "0"), (8, 2, "0,1"), (26, 1, "1"), (26, 1, "0"), (26, 2, "0,1")]
    # G3 纯CPU
    # for (NCPU, NCUDA, TCUDA) in [(8, 0, ""), (26, 0, "")]
    # G4 其他硬件环境
    # for (NCPU, NCUDA, TCUDA) in [(8, 1, "1"), (8, 1, "0"), (8, 2, "0,1"), (26, 1, "1"), (26, 1, "0"), (26, 2, "0,1")]
]
scheduling_params = [
    (SCHED, PRIOR)
    for (SCHED, PRIOR) in [("dmda", 0), ("dmdap", 0), ("dmdap", 1), ("dmdap", 2)]
]

# 执行实验和保存结果
output_file_path = os.path.join(res_dir, res_name)

with open(output_file_path, "w") as f:
    header = "{:<8}\t{:<5}\t{:<5}\t{:<7}\t{:<6}\t{:<5}".format("NBLOCKS", "NCPU", "NCUDA", "TCUDA", "SCHED", "PRIOR")
    for i in range(N_skip, N):
        header += f"\tGFLOPS_{i}"
    header += "\tGFLOPS\n"
    f.write(header)

for nblocks in range(2, NBLOCKS+2, 2):
    for (NCPU, NCUDA, TCUDA) in hardware_env:
        for (SCHED, PRIOR) in scheduling_params:
            # 设置环境变量
            os.environ["STARPU_NCPU"] = str(NCPU)
            os.environ["STARPU_NCUDA"] = str(NCUDA)
            os.environ["CUDA_VISIBLE_DEVICES"] = TCUDA
            os.environ["STARPU_SCHED"] = SCHED
            os.environ["STARPU_HOSTNAME"] = f"{date}_{NCPU}_{NCUDA}_{TCUDA}_{SCHED}"
            
            os.system("rm /root/.starpu/sampling/codelets/45/*")
            os.system(f"rm /root/.starpu/sampling/bus/*")

            # 初始化结果列表
            results = []

            # 执行 N 次实验
            for _ in range(N):
                # 执行命令
                cmd = f"bash {task_script} -size {nblocks * 960} -nblocks {nblocks} -priority_attribution_p {PRIOR}"
                output = subprocess.check_output(cmd, shell=True, text=True)

                # 提取 GFLOPS 数据
                readGFlops = 0
                for line in output.splitlines():
                    if readGFlops == 1:
                        # 提取 GFLOPS 数据
                        gflops1 = float(line.split()[-1])
                        break
                    if "GFlop/s" in line:
                        # print(line)
                        # gflops1 = float(line.split()[-1])
                        readGFlops = 1
                    print(line, end='')  # 显示输出到终端
                # print(gflops1)
                
                # 将结果添加到列表
                results.append(gflops1)  

            # 丢弃第一次实验的结果
            results = results[N_skip:]

            # 计算 N-N_skip 次结果的均值
            if len(results) == N - N_skip:
                avg_gflops = sum(results) / (N - N_skip)
            else:
                avg_gflops = 0.0

            # 保存结果到文件
            with open(output_file_path, "a") as f:
                line = "{:<8}\t{:<5}\t{:<5}\t{:<7}\t{:<6}\t{:<5}\t".format(nblocks, NCPU, NCUDA, TCUDA, SCHED, PRIOR)
                for gflops in results:
                    line += "{:<8.1f}\t".format(gflops)
                line += "{:<8.1f}\n".format(avg_gflops)
                f.write(line)
            
            os.system("rm /root/.starpu/sampling/codelets/45/*")
            os.system(f"rm /root/.starpu/sampling/bus/*")

print(f"Results saved to {output_file_path}")
