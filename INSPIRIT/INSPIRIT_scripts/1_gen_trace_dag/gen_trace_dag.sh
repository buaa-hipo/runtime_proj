#!/bin/bash

# 1. 参数设置
# task_name="cholesky"
# task_directory="/root/starpu_trace/examples/{$task_name}"
# task_script="cholesky_implicit"

# $1 $2 $3
# ./gen_trace_dag {task_name} {task_path} {task_script}
# ./gen_trace_dag.sh "cholesky" "/root/starpu_trace_view/examples/cholesky" "cholesky_implicit"
task_name=$1
task_directory=$2
task_script=$3
NBLOCKS=$4
cur_path=$(readlink -f "$(dirname "$0")")

    # sche
[ -z "$STARPU_SCHED" ] || STARPU_SCHEDS="$STARPU_SCHED"
[ -n "$STARPU_SCHEDS" ] || STARPU_SCHEDS="dmda"

[ -n "$STARPU_HOSTNAME" ] || export STARPU_HOSTNAME=mirage
unset MALLOC_PERTURB_

    # size
INCR=2
STOP=$NBLOCKS

# 2. 切换目标目录task_directory，生成prof_file__0
cd "$task_directory" || exit 1

(
    # 记录trace
export STARPU_FXT_TRACE=1
    # 执行脚本
for size in $(seq 2 $INCR $STOP); do
  for STARPU_SCHED in $STARPU_SCHEDS; do
    echo -n "$((size * 960))"
    echo -n "   $STARPU_SCHED"
    export STARPU_SCHED
    GFLOPS=$(bash $task_script -size $((size * 960)) -nblocks $size 2> /dev/null | grep -v GFlop/s | cut -d '	' -f 3)
    [ -n "$GFLOPS" ] || GFLOPS='""'
    echo -n "	$GFLOPS"
    echo
    # 输出文件重命名
    if [ -f "/tmp/prof_file__0" ]; then
      if [ ! -d "$cur_path/$task_name/prof_file" ]; then
        mkdir "$cur_path/$task_name/prof_file"
      fi
      mv "/tmp/prof_file__0" "$cur_path/$task_name/prof_file/prof_file_$((size * 960))_$STARPU_SCHED"
    fi
  done
done
)

# 3. 返回原目录/root/PriorStarPU，转化为dag.dot
cd - || exit 1

for file in "$task_name/prof_file"/*; do
  # 判断是普通文件
  if [ -f "$file" ]; then
  filename=$(basename "$file")
    # 判断以prof_file_起始
    if [[ "$filename" == "prof_file_"* ]]; then
      cp "$task_name/prof_file/$filename" "prof_file__0"
      starpu_fxt_tool -i "prof_file__0"
      if [ ! -d "$cur_path/$task_name/dot" ]; then
        mkdir "$cur_path/$task_name/dot"
      fi
      mv "dag.dot" "$cur_path/$task_name/dot/dot_$filename"
    fi
  fi
done

rm "activity.data" "comms.rec" "data.rec" "distrib.data" "paje.trace" "prof_file__0" "sched_tasks.rec" "tasks.rec" "trace.html" "trace.rec"


# 4. 执行$task_name/dot2dict.py文件，将dag.dot转换为dag（本脚本最终结果）
for file in "$task_name/dot"/*; do
  # 判断是普通文件
  if [ -f "$file" ]; then
  filename=$(basename "$file")
    # 判断以prof_file_起始
    if [[ "$filename" == "dot_prof_file_"* ]]; then
      cp "$task_name/dot/$filename" "$task_name/dag.dot"
      python3 "$task_name/dot2dict.py"
      if [ ! -d "$cur_path/$task_name/dag" ]; then
        mkdir "$cur_path/$task_name/dag"
      fi
      mv "dag.txt" "$cur_path/$task_name/dag/dag_$filename.txt"
    fi
  fi
done

rm "$task_name/dag.dot"

# 5. 结束脚本
exit 0
