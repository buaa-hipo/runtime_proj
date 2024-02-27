POTRF = []
TRSM = []
SYRK = []
GEMM = []
tmp = []
import  os

cur_path = os.getcwd()

with open(f"{cur_path}/cholesky/dag.dot", "r") as f:
    for line in f:
        if line.strip().startswith("\"task_"):
            if "->" not in line:
                line = line.strip().strip("\"")
                src = line.split("\" ")[0]
                src = int(src.lstrip("task_"))
                if src in tmp:
                    continue
                else:
                    tmp.append(src)
                    _, dst = line.split("\" ")
                    _, dst_label, _, _ = dst.split(", ")
                    dst_label = dst_label.lstrip("label=\"").rstrip("\"")
                    if dst_label == "POTRF":
                        POTRF.append(src)
                    elif dst_label == "TRSM":
                        TRSM.append(src)
                    elif dst_label == "SYRK":
                        SYRK.append(src)
                    elif dst_label == "GEMM":
                        GEMM.append(src)

f.close()

# print("POTRF: ", POTRF)
# print("TRSM: ", TRSM)
# print("SYRK: ", SYRK)
# print("GEMM: ", GEMM)

dict = {}
tmp = []
# 打开文件dag.dot
with open(f"{cur_path}/cholesky/dag.dot", "r") as f:
    for line in f:
        if line.strip().startswith("\"task_"):
            if "->" in line:
                line = line.strip().strip("\"")
                src, dst = line.split("\"->\"")
                src = int(src.lstrip("task_"))
                dst = int(dst.lstrip("task_"))
                # print(src, dst)
                if src in POTRF or src in TRSM or src in SYRK or src in GEMM:
                    if dst in POTRF or dst in TRSM or dst in SYRK or dst in GEMM:
                        if src not in dict:
                            dict[src] = (dst, )
                        else:
                            dict[src] = dict[src] + (dst, )
            else:
                src = line.split("\" ")[0]
                src = int(src.lstrip("\t \"task_"))
                if src in tmp:
                    continue
                else:
                    tmp.append(src)
                    line = line.strip().strip("\"")
                    _, dst = line.split("\" ")
                    # print(src, dst)
                    if src in POTRF or src in TRSM or src in SYRK or src in GEMM:
                        if src not in dict:
                            dict[src] = ()       
# print(dict)
# for i in dict:
#     print(i, dict[i])

cnt = 0
dict_table = {}
# 对字典进行重编号 从零开始
new_dict = {}

for i in dict:
    if i not in dict_table:
        dict_table[i] = cnt
        cnt += 1
    new_dict[dict_table[i]] = ()
    for j in dict[i]:
        if j not in dict_table:
            dict_table[j] = cnt
            cnt += 1
        new_dict[dict_table[i]] = new_dict[dict_table[i]] + (dict_table[j], )
# for i in new_dict:
#     print(f"{i}:{new_dict[i]},")

new_POTRF = []
new_TRSM = []
new_SYRK = []
new_GEMM = []

for i in POTRF:
    new_POTRF.append(dict_table[i])
for i in TRSM:
    new_TRSM.append(dict_table[i])
for i in SYRK:
    new_SYRK.append(dict_table[i])
for i in GEMM:
    new_GEMM.append(dict_table[i])
# print("new_POTRF: ", new_POTRF)
# print("new_TRSM: ", new_TRSM)
# print("new_SYRK: ", new_SYRK)
# print("new_GEMM: ", new_GEMM)

# 写入结果文件
with open("dag.txt", "w") as file:
    file.write(f"POTRF: {new_POTRF}\n")
    file.write(f"TRSM: {new_TRSM}\n")
    file.write(f"SYRK: {new_SYRK}\n")
    file.write(f"GEMM: {new_GEMM}\n")
    file.write(f"\n")    
    for i in new_dict:
        file.write(f"{i}:{new_dict[i]},\n")