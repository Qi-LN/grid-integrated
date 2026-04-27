import argparse
import json
from pathlib import Path

import dgl
import numpy as np
import torch


DATASET_INFO = {
    "products": 2449029,
    "paper100m": 111059956,
    "com-friendster": 65608366,
    "ukunion": 133633040,
    "uk2014": 787801471,
    "clueweb": 955207488,
}

# 从二进制文件中读取
def load_csr(dataset_dir):
    indptr = np.fromfile(dataset_dir / "edge_src", dtype=np.int64)
    indices = np.fromfile(dataset_dir / "edge_dst", dtype=np.int32)
    return indptr, indices


def build_dgl_graph(indptr, indices, num_nodes):
    indptr_t = torch.from_numpy(indptr)
    indices_t = torch.from_numpy(indices.astype(np.int64, copy=False))
    eids_t = torch.arange(indices_t.numel(), dtype=torch.int64)
    return dgl.graph(("csr", (indptr_t, indices_t, eids_t)), num_nodes=num_nodes)


# 根据 parts 里记录的节点分区结果，把每个 partition 拥有的节点收集起来
def assign_owner_nodes(parts, num_nodes):
    owners = [[] for _ in range(num_nodes)]
    for node_id, part_id in enumerate(parts.tolist()):
        owners[int(part_id)].append(node_id)
    return [np.asarray(nodes, dtype=np.int32) for nodes in owners]

# 把当前计算节点 node_rank 拥有的 owner nodes，平均切分给本节点上的多张 GPU
def split_owner_roots_to_gpus(owner_nodes, gpus_per_node):
    splits = []
    total = owner_nodes.size
    for gpu_id in range(gpus_per_node):
        start = (gpu_id * total) // gpus_per_node
        end = ((gpu_id + 1) * total) // gpus_per_node
        splits.append(owner_nodes[start:end].astype(np.int32, copy=False))
    return splits


def compute_node_halo(indptr, indices, owner_nodes, owner_part):
    halo = set()
    for root in owner_nodes:
        start = int(indptr[root])
        end = int(indptr[root + 1])
        for nbr in indices[start:end]:
            nbr = int(nbr)
            if owner_part[nbr] != owner_part[root]:
                halo.add(nbr)
    return np.asarray(sorted(halo), dtype=np.int32)


def assign_halo_to_gpus(halo_nodes, gpus_per_node):
    splits = [[] for _ in range(gpus_per_node)]
    for node_id in halo_nodes.tolist():
        splits[node_id % gpus_per_node].append(node_id)
    return [np.asarray(nodes, dtype=np.int32) for nodes in splits]


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def owner_batch_index_map(owner_gpu_nodes, batch_size):
    mapping = {}
    for gpu_id, roots in enumerate(owner_gpu_nodes):
        for local_idx, node_id in enumerate(roots.tolist()):
            step = local_idx // batch_size
            batch_idx = local_idx % batch_size
            mapping[int(node_id)] = (gpu_id, step, batch_idx)
    return mapping


def append_send(plan, step, dst_rank, batch_idx):
    while len(plan) <= step:
        plan.append({"targets": []})
    targets = plan[step].setdefault("targets", [])
    for entry in targets:
        if entry["dst"] == dst_rank:
            entry["batch_indices"].append(batch_idx)
            return
    targets.append({"dst": dst_rank, "batch_indices": [batch_idx]})


def append_recv(plan, step, src_rank, local_offset):
    while len(plan) <= step:
        plan.append({"sources": []})
    sources = plan[step].setdefault("sources", [])
    for entry in sources:
        if entry["src"] == src_rank:
            entry["recv_local_offsets"].append(local_offset)
            entry["count"] = len(entry["recv_local_offsets"])
            return
    sources.append({"src": src_rank, "recv_local_offsets": [local_offset], "count": 1})


def write_bin(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(np.int32, copy=False).tofile(path)


def main():
    parser = argparse.ArgumentParser("Generate multi-node halo metadata for Legion coordinate inference.")
    parser.add_argument("--dataset_path", type=str, default="dataset")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--gpus_per_node", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--partition_path", type=str, default="")
    parser.add_argument("--save_partition", action="store_true")
    parser.add_argument("--metis_seed", type=int, default=0)
    args = parser.parse_args()

    if args.dataset_name not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    if args.num_nodes <= 0 or args.gpus_per_node <= 0 or args.batch_size <= 0:
        raise ValueError("num_nodes, gpus_per_node and batch_size must be positive")

    # csr文件
    dataset_dir = Path(args.dataset_path) / args.dataset_name
    # 未指定输出路径则放在数据集目录下的halo_partition子目录
    output_dir = Path(args.output_path) if args.output_path else dataset_dir / "halo_partition"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = DATASET_INFO[args.dataset_name]
    # 读取csr
    indptr, indices = load_csr(dataset_dir)
    if indptr.size != num_nodes + 1:
        raise ValueError(f"edge_src length {indptr.size} does not match num_nodes + 1 ({num_nodes + 1})")

    # 如果已有分区文件
    if args.partition_path:
        parts = np.fromfile(args.partition_path, dtype=np.int32)
        if parts.size != num_nodes:
            raise ValueError(f"partition length {parts.size} does not match node count {num_nodes}")
    # 如果没传分区文件，就直接用 DGL 的 METIS 做图划分：
    else:
        torch.manual_seed(args.metis_seed)
        graph = build_dgl_graph(indptr, indices, num_nodes)
        parts_t = dgl.metis_partition_assignment(graph, args.num_nodes)
        parts = parts_t.cpu().numpy().astype(np.int32, copy=False)
        if args.save_partition:
            # parts长度等同vertex数量，标注每个节点的分区号
            parts.tofile(output_dir / "node_partition.bin")

    # 把每个gpu的节点都汇聚起来
    owner_nodes_by_node = assign_owner_nodes(parts, args.num_nodes)
    owner_gpu = []
    halo_gpu = []
    infer_steps = 0
    # 遍历每个节点
    for node_rank in range(args.num_nodes):
        owner_splits = split_owner_roots_to_gpus(owner_nodes_by_node[node_rank], args.gpus_per_node)
        owner_gpu.append(owner_splits)
        for roots in owner_splits:
            infer_steps = max(infer_steps, ceil_div(roots.size, args.batch_size))
        halo_nodes = compute_node_halo(indptr, indices, owner_nodes_by_node[node_rank], parts)
        halo_gpu.append(assign_halo_to_gpus(halo_nodes, args.gpus_per_node))

    (output_dir / "infer_steps.txt").write_text(f"{infer_steps}\n", encoding="utf-8")

    global_rank_count = args.num_nodes * args.gpus_per_node
    rank_plans = [{"send": [{} for _ in range(infer_steps)], "recv": [{} for _ in range(infer_steps + 1)]}
                  for _ in range(global_rank_count)]

    owner_maps = [owner_batch_index_map(owner_gpu[node_rank], args.batch_size)
                  for node_rank in range(args.num_nodes)]

    for node_rank in range(args.num_nodes):
        node_dir = output_dir / f"node_{node_rank}"
        node_dir.mkdir(parents=True, exist_ok=True)
        for gpu_id in range(args.gpus_per_node):
            write_bin(node_dir / f"gpu_{gpu_id}_owner_roots.bin", owner_gpu[node_rank][gpu_id])
            write_bin(node_dir / f"gpu_{gpu_id}_halo_nodes.bin", halo_gpu[node_rank][gpu_id])

    for dst_node in range(args.num_nodes):
        for dst_gpu in range(args.gpus_per_node):
            dst_rank = dst_node * args.gpus_per_node + dst_gpu
            owner_count = owner_gpu[dst_node][dst_gpu].size
            for halo_idx, node_id in enumerate(halo_gpu[dst_node][dst_gpu].tolist()):
                src_node = int(parts[node_id])
                if src_node == dst_node:
                    continue
                src_gpu, step, batch_idx = owner_maps[src_node][node_id]
                src_rank = src_node * args.gpus_per_node + src_gpu
                local_offset = int(owner_count + halo_idx)
                append_send(rank_plans[src_rank]["send"], step, dst_rank, int(batch_idx))
                append_recv(rank_plans[dst_rank]["recv"], step + 1, src_rank, local_offset)

    for rank, plan in enumerate(rank_plans):
        with open(output_dir / f"rank_{rank}.json", "w", encoding="utf-8") as file:
            json.dump(plan, file, separators=(",", ":"))

    summary = {
        "dataset_name": args.dataset_name,
        "num_nodes": args.num_nodes,
        "gpus_per_node": args.gpus_per_node,
        "batch_size": args.batch_size,
        "infer_steps": infer_steps,
        "owner_counts": [[int(x.size) for x in node] for node in owner_gpu],
        "halo_counts": [[int(x.size) for x in node] for node in halo_gpu],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Wrote halo metadata to {output_dir}")
    print(f"Infer steps: {infer_steps}")


if __name__ == "__main__":
    main()
