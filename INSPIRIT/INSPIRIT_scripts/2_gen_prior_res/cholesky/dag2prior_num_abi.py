# 计算每个节点time在bound范围内的后继节点数
import argparse
import math

# 依赖图
graph = {}
# 类型_节点
type_nodes = {}
# 节点_类型
node2type = {}
# 节点_后继节点
node2children = {}
# 节点_k步_后继节点
node2kchildren = {}
# 节点_符合条件_后继节点
node2timechildren = {}
# 节点_启发能力
node2inspireAbi = {}
# 节点_启发效率
node2inspireEfi = {}
# 节点_优先级
node2prior = {}
# 类型_costModel
bound = 218346.88
type2time = {
    # 10.917,344s
    # 10917344us
    "POTRF": 1.853913e+04,
    "TRSM": 5.624431e+05,
    "SYRK": 2.051288e+02,
    "GEMM": 1.475663e+02
}

import  os

cur_path = os.getcwd()

# 读取dag.txt生成图数据和节点类型数据
def load_dag():
    # 打开文本文件并逐行读取数据
    with open(f'{cur_path}/cholesky/dag.txt', 'r') as file:
        lines = file.readlines()

    # 遍历每一行数据
    for line in lines:
        # 去除前导和尾随空白字符
        line = line.strip()
        
        # 如果行为空，跳过
        if not line:
            continue
        # 检查行是否表示任务类型
        if line.startswith('POTRF:') or line.startswith('TRSM:') or line.startswith('SYRK:') or line.startswith('GEMM:'):
            type, node_str = line.split(':')
            nodes = [int(node.strip()) for node in node_str.strip(' [').strip(']').split(',') if node.strip()]
            type_nodes[type] = nodes
        else:
            # 否则，行表示图结构
            node, children_str = line.split(':')
            node = int(node)
            # 使用列表推导来过滤并转换子节点
            children = [int(child.strip()) for child in children_str.strip('(').strip('),').split(',') if child.strip()]
            graph[node] = children

# 将节点类型数据转换为对给定节点的类型索引
def get_node_type():
    # 遍历 type2node 字典的键值对
    for type, nodes in type_nodes.items():
        # 遍历每个节点并将其与任务类型关联
        for node in nodes:
            node2type[node] = type

# 传入参数k计算节点的启发能力
def get_inspire_abi(k):
    # def calculate_k_steps(node, k):
    #     if k == 0:
    #         return [node]  # 当k为0时，节点本身就是一个后k步节点
    #     if node not in graph:
    #         return []  # 如果节点不存在于图中，返回空列表
    #     result = []
    #     for child in graph[node]:
    #         child_k_steps = calculate_k_steps(child, k - 1)
    #         result.extend(child_k_steps)
    #     return result
    # # 计算每个节点的后k步节点数
    # for node in graph:
    #     node2kchildren[node] = []
    #     for i in range(1, k+1):
    #         node2kchildren[node].extend(calculate_k_steps(node, i))
    #     node2kchildren[node] = list(set(node2kchildren[node]))
    
    from collections import deque
    def get_all_successors(graph):
        successors = {}
        for node in graph:
            visited = set()
            queue = deque([node])
            successors[node] = []
            while queue:
                current_node = queue.popleft()
                visited.add(current_node)
                for neighbor in graph.get(current_node, []):
                    successors[node].append(neighbor)
                    if neighbor not in visited:
                        queue.append(neighbor)
            # 去重后的后继节点列表
            successors[node] = list(set(successors[node]))
        return successors
    # 计算每个节点的所有后继节点数
    node2children = {}
    node2children.update(get_all_successors(graph))
    node2children = dict(sorted(node2children.items()))
    
    # for node, ability in node2children.items():
    #     print(f"Node {node} Type {node2type[node]}: len {len(ability)} nodes in all steps")
    
    for node in graph:
        node2inspireAbi[node] = len(node2children[node])
    # for node in graph:
    #     node2inspireAbi[node] = len(node2kchildren[node])

# 传入参数k, type计算节点的启发效率
# def get_inspire_efi(k, type):
#     for node in graph:
#         children = node2children[node]
#         # 根据type获得node_list的efi
#         efi = 0
#         for child in children:
#             type = node2type[child]
#             # 映射关系 TODO
#             time = type2time[type]
#             efi += time
#         node2inspireEfi[node] = efi

def get_inspire_efi(bound):
    def calculate_time_steps(node, cur):
        result = []
        for child in graph[node]:
            time = type2time[node2type[child]]
            if cur + time <= bound:
                result.append(child)
                result.extend(calculate_time_steps(child, cur+time))
        return result

    # 计算每个节点timebound内的后k步节点数
    for node in graph:
        node2timechildren[node] = []
        node2timechildren[node].extend(calculate_time_steps(node, 0))
        node2timechildren[node] = list(set(node2timechildren[node]))
        # node2timechildren[node] = len(node2timechildren[node])
    for node in graph:
        children = node2timechildren[node]
        node2inspireEfi[node] = len(children)
        # print(f"Node {node} Type {node2type[node]}: len {node2inspireEfi[node]} efficient")

    
# 根据启发能力和启发效率综合生成优先级
def gen_prior():
    # 根据启发能力和启发效率生成优先级 TODO
    alpha = 1 # 0.018
    beta = 0
    for node in graph:
        node2prior[node] = math.floor(alpha * node2inspireAbi[node])
        # node2prior[node] = math.floor(alpha * node2inspireAbi[node] + beta * node2inspireEfi[node])
        # node2prior[node] = beta * node2inspireEfi[node]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, help="对节点的启发能力考虑多少", default=1)
    parser.add_argument("--type", type=str, help="对节点的启发效率考虑多少", default="sum")
    
    args = parser.parse_args()
    k = args.k
    type = args.type
    # print(f"k: {k} type: {type}")

    # 生成graph
    load_dag()
    # print("Type nodes:")
    # for type, nodes in type_nodes.items():
    #     print(f"{type}: {nodes}")
    # print("Graph:")
    graph = dict(sorted(graph.items()))
    # for node, children in graph.items():
    #     print(f"{node}: {children},")
    
    # 生成node2type
    get_node_type()
    # print("Node2type:")
    # for node, task_type in node2type.items():
    #     print(f"Node {node}: {task_type}")
    
    # 启发能力
    get_inspire_abi(k)
    # node2inspireAbi = dict(sorted(node2inspireAbi.items()))
    # for node, ability in node2inspireAbi.items():
    #     print(f"Node {node}: {ability} nodes in {k} steps")
    # # node2kchildren = dict(sorted(node2kchildren.items()))
    # for node, ability in node2kchildren.items():
    #     print(f"Node {node} Type {node2type[node]}: len {len(ability)} {ability} nodes in {k} steps")
    # node2children = dict(sorted(node2children.items()))
    # for node, ability in node2children.items():
    #     print(f"Node {node} Type {node2type[node]}: len {len(ability)} nodes in all steps")
    #     # print(f"Node {node} Type {node2type[node]}: len {len(ability)} {ability} nodes in all steps")
    
    # 启发效率
    # get_inspire_efi(k, type)
    # get_inspire_efi(bound)
    # for node, children in node2timechildren.items():
    #     print(f"Node {node} Type {node2type[node]}: len {len(children)} {children} in time bound")
    # node2inspireEfi = dict(sorted(node2inspireEfi.items()))
    # for node, efficient in node2inspireEfi.items():
    #     print(f"Node {node} Type {node2type[node]}: {efficient} efficient")
    
    # 生成节点对应的优先级
    gen_prior()
    # print(f"config: k_{k}_type_{type}")
    # for node, prior in node2prior.items():
    #     print(f"Node {node}: {prior}")
        
    # 按字典的键从小到大排序
    new_dict = {}
    for k, v in node2prior.items():
        new_dict[int(k)] = v
    node2prior = new_dict
    sorted_priors = dict(sorted(node2prior.items()))


    # 获取排序后的值数组
    priors = list(sorted_priors.values())

    # 将列表转换为字符串，去掉空格，然后再转回列表
    formatted_priors = str(priors).replace(" ", "")
    
    print(formatted_priors)
    