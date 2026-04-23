import os
import argparse
import subprocess
import networkx as nx
import math


def parse_topo_output(output):
    connections = []
    lines = output.splitlines()
    gpu_lines = [line for line in lines if line.startswith("GPU")]
    for i, line in enumerate(gpu_lines):
        elements = line.split()
        for j, elem in enumerate(elements[1:], start=0):
            if elem.startswith("NV"):
                connections.append((i, j))
    return connections


def get_nvlink_topology():
    result = subprocess.run(['nvidia-smi', 'topo', '-m'], stdout=subprocess.PIPE, text=True)
    connections = parse_topo_output(result.stdout)
    return connections


def find_largest_fully_connected_group(G):
    cliques = list(nx.find_cliques(G))
    max_size = max(len(clique) for clique in cliques) if cliques else 1
    max_cliques = [clique for clique in cliques if len(clique) == max_size]
    return max_size, max_cliques

# 从 partition_meta_path 目录里读取 infer_steps.txt 文件，并把其中的数字作为“推理步数覆盖值”返回。
def infer_step_override_from_meta(partition_meta_path):
    # 路径无效返回-1
    if not partition_meta_path:
        return -1
    # 文件不存在
    infer_steps_path = os.path.join(partition_meta_path, 'infer_steps.txt')
    if not os.path.exists(infer_steps_path):
        return -1
    with open(infer_steps_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    return int(content) if content else -1


def Run(args):

    if args.dataset_name == "products":
        path = args.dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 100
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
    elif args.dataset_name == "paper100m":
        path = args.dataset_path + "/paper100m/"
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 1207179
        valid_set_num = 125265
        test_set_num = 214338
    elif args.dataset_name == "com-friendster":
        path = args.dataset_path + "/com-friendster/"
        vertices_num = 65608366
        edges_num = 1806067135
        features_dim = 256
        train_set_num = 6560836
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "ukunion":
        path = args.dataset_path + "/ukunion/"
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 256
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "uk2014":
        path = args.dataset_path + "/uk2014/"
        vertices_num = 787801471
        edges_num = 47284178505
        features_dim = 128
        train_set_num = 78780147
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "clueweb":
        path = args.dataset_path + "/clueweb/"
        vertices_num = 955207488
        edges_num = 42574107469
        features_dim = 128
        train_set_num = 95520748
        valid_set_num = 100000
        test_set_num = 100000
    else:
        print("invalid dataset path")
        return

    serve_mode = 1 if args.mode == "infer" else 0
    rootset_path = args.rootset_path if args.rootset_path else "-"
    partition_meta_path = args.partition_meta_path if args.partition_meta_path else "-"
    # 获取推理步数值
    infer_step_override = infer_step_override_from_meta(partition_meta_path)
    with open("meta_config", "w", encoding='utf-8') as file:
        file.write(
            "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                path,
                args.train_batch_size,
                vertices_num,
                edges_num,
                features_dim,
                train_set_num,
                valid_set_num,
                test_set_num,
                args.cache_memory,
                args.epoch,
                serve_mode,
                rootset_path,
                partition_meta_path,
                args.node_rank,
                args.num_nodes,
                args.local_gpu_number,
                infer_step_override,
            )
        )

    gpu_number = args.gpu_number

    if args.usenvlink == 1 and args.mode != 'infer':
        connections = get_nvlink_topology()
        G = nx.Graph()
        G.add_edges_from(connections)
        group_size, fully_connected_groups = find_largest_fully_connected_group(G)
        if fully_connected_groups or group_size == 1:
            print(f"NVLink clique size: {group_size}, Number of NVLink cliques: {int(gpu_number/group_size)}")
        cache_agg_mode = math.log2(group_size)
    else:
        cache_agg_mode = 0

    current_file_path = os.path.abspath(__file__)
    Legion_home = os.path.dirname(current_file_path)
    server_path = os.path.join(Legion_home, "sampling_server/build/bin/sampling_server {} {}").format(gpu_number, cache_agg_mode)
    os.system(server_path)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser("Legion Server.")
    argparser.add_argument('--dataset_path', type=str, default="./dataset")
    argparser.add_argument('--dataset_name', type=str, default="ukunion")
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--fanout', type=list, default=[25, 10])
    argparser.add_argument('--gpu_number', type=int, default=2)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--cache_memory', type=int, default=38000000)
    argparser.add_argument('--usenvlink', type=int, default=1)
    argparser.add_argument('--mode', type=str, choices=['train', 'infer'], default='train')
    argparser.add_argument('--rootset_path', type=str, default='')
    argparser.add_argument('--partition_meta_path', type=str, default='')
    argparser.add_argument('--node_rank', type=int, default=0)
    argparser.add_argument('--num_nodes', type=int, default=1)
    argparser.add_argument('--local_gpu_number', type=int, default=2)
    args = argparser.parse_args()

    Run(args)
