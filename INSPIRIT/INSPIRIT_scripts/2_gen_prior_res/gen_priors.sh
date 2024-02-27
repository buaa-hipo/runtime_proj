

# 1. 参数设置
# task_name="cholesky"
# task_directory="/root/INSPIRIT/examples/cholesky"
# task_script="cholesky_implicit"

# $1 $2 $3
# ./gen_trace_dag {task_name} {task_path} {task_script}
# ./gen_trace_dag.sh "cholesky" "/root/starpu_trace_view/examples/cholesky" "cholesky_implicit"
task_name=$1
task_directory=$2
task_script=$3
NBLOCKS=$4
cur_path=$(readlink -f "$(dirname "$0")")

    # size
START=2
INCR=2
STOP=$NBLOCKS

# 2. 生成不同参数表现的优先级
for size in $(seq $START $INCR $STOP); do
    echo "Now run $size"
    cp "$cur_path/../1_gen_trace_dag/$task_name/dag/dag_dot_prof_file_$(($size * 960))_dmda.txt" "$cur_path/$task_name/dag.txt"
    if [ ! -d "$cur_path/$task_name/priors_abi" ]; then
        mkdir "$cur_path/$task_name/priors_abi"
    fi
    python3 "$cur_path/$task_name/dag2prior_num_abi.py" > "$cur_path/$task_name/priors_abi/$(($size * 960)).txt"
    if [ ! -d "$cur_path/$task_name/priors_efi" ]; then
        mkdir "$cur_path/$task_name/priors_efi"
    fi
    python3 "$cur_path/$task_name/dag2prior_num_efi.py" > "$cur_path/$task_name/priors_efi/$(($size * 960)).txt"
done
rm "$cur_path/$task_name/dag.txt"
# cd - || exit 1
