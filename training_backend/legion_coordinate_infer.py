import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import dgl
import ipc_service
import torch
import torch.distributed as dist
from dgl.heterograph import DGLBlock
from dgl.nn.pytorch import SAGEConv


def create_dgl_block(src, dst, num_src_nodes, num_dst_nodes):
    gidx = dgl.heterograph_index.create_unitgraph_from_coo(
        2,
        num_src_nodes,
        num_dst_nodes,
        src,
        dst,
        "coo",
        row_sorted=True,
    )
    return DGLBlock(gidx, (["_N"], ["_N"]), ["_E"])


class CoordinateUpdater(torch.nn.Module):
    def __init__(self, coord_dim, hidden_dim):
        super().__init__()
        self.conv = SAGEConv(coord_dim, hidden_dim, "mean")
        self.proj = torch.nn.Linear(hidden_dim, coord_dim)

    def forward(self, block, features):
        hidden = self.conv(block, features)
        return self.proj(hidden)


# 从环境变量里推断当前进程的分布式身份信息
def infer_rank_info() -> Tuple[int, int, int]:
    # 当前进程的全局编号
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    # 本节点内的编号
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    # 总共启动了多少个进程
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    return rank, local_rank, world_size


# 如果是多进程，就初始化 torch.distributed
def init_dist_if_needed(rank: int, world_size: int):
    if world_size <= 1 or dist.is_initialized():
        return
    dist.init_process_group(
        backend=os.environ.get("LEGION_DIST_BACKEND", "gloo"),
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def finalize_dist_if_needed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# 负责给当前 rank 加载一份“按 step 划分的通信计划”.(告诉当前进程，每一步要send什么和recv什么)
def load_rank_plan(meta_path: str, rank: int, infer_steps: int, require_plan: bool) -> Dict[str, Any]:
    # 多机模式无meta文件直接报错
    if not meta_path:
        if require_plan:
            raise RuntimeError("partition_meta_path is required for multi-node inference")
        # 无meta但是单机则返回空计划
        return {"send": [{} for _ in range(infer_steps)], "recv": [{} for _ in range(infer_steps + 1)]}
    plan_path = os.path.join(meta_path, f"rank_{rank}.json")
    # 多机无文件则报错
    if not os.path.exists(plan_path):
        if require_plan:
            raise RuntimeError(f"missing rank plan file: {plan_path}")
        return {"send": [{} for _ in range(infer_steps)], "recv": [{} for _ in range(infer_steps + 1)]}
    # 读取当前rank的json计划文件
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    # 列表中存储的是字典
    send_plan = plan.get("send", [])
    recv_plan = plan.get("recv", [])
    if len(send_plan) < infer_steps:
        raise RuntimeError("plan length less than infer step.")
        # 补充空计划
        # send_plan = send_plan + [{} for _ in range(infer_steps - len(send_plan))]
    if len(recv_plan) < infer_steps + 1:
        raise RuntimeError("plan length less than infer step.")
        # recv_plan = recv_plan + [{} for _ in range(infer_steps + 1 - len(recv_plan))]
    return {"send": send_plan, "recv": recv_plan}
    # send_plan = [
    #     {
    #         "targets": [
    #             {"dst": 1, "batch_indices": [0, 3, 7]},
    #             {"dst": 4, "batch_indices": [2, 5]}
    #         ]
    #     },
    #     {
    #         "targets": [
    #             {"dst": 1, "batch_indices": [1, 6]},
    #             {"dst": 3, "batch_indices": [0, 4, 8]}
    #         ]
    #     },
    #     {
    #         "targets": [
    #             {"dst": 4, "batch_indices": [2]}
    #         ]
    #     }
    # ]
    # 访问方式：plan["send"][step]["targets"][length]["dst"/"batch_indices"]

# 把某个 step 的计划项里，按key取出对应的列表；如果没有，就返回空列表。
def normalize_entries(step_entry: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    if not step_entry:
        return []
    return step_entry.get(key, [])


# 从 GPU 上的 updated_coords 里按索引抽取一部分行，把它们拷贝到一块“可固定页”的 CPU 内存里，作为后续 dist.isend(...) 的发送 payload。
def stage_send_payload(updated_coords: torch.Tensor, gpu_indices: torch.Tensor) -> torch.Tensor:
    # 无索引则返回空向量
    if gpu_indices.numel() == 0:
        return torch.empty((0, updated_coords.size(1)), dtype=updated_coords.dtype)
    # 从 updated_coords 中挑出 gpu_indices 指定的那些行
    gathered = updated_coords.index_select(0, gpu_indices)
    # cpu上分配锁页内存
    cpu_payload = torch.empty(
        (gathered.size(0), gathered.size(1)),
        dtype=gathered.dtype,
        device="cpu",
        pin_memory=True,
    )
    # 复制到cpu
    cpu_payload.copy_(gathered, non_blocking=True)
    torch.cuda.current_stream(updated_coords.device).synchronize()
    return cpu_payload


# 把异步接收等完，然后把收到的 CPU 坐标数据搬到 GPU 上，再按 recv_local_offsets 写回到本地坐标存储里
def apply_received_updates(device, pending_recvs):
    # 遍历各src_rank
    for entry in pending_recvs:
        req = entry["req"]
        if req is not None:
            req.wait()
        # 本地shard写回位置
        recv_local_offsets = entry["recv_local_offsets"]
        # wait后引用已获取通信数据，可以取出进行dma了
        cpu_tensor = entry["cpu_tensor"]
        if cpu_tensor.numel() == 0:
            continue
        gpu_tensor = cpu_tensor.to(device=device, non_blocking=True)
        # TODO:确定这里异步是否有意义？
        ipc_service.update_coordinates(recv_local_offsets, gpu_tensor.contiguous())
    pending_recvs.clear()


def post_recvs_for_step(step_entry, coord_dim, device):
    pending = []
    # 依次遍历来自各个src_rank的传输
    for source in normalize_entries(step_entry, "sources"):
        src = int(source["src"])
        # 本地写回位置
        recv_local_offsets = torch.tensor(
            source.get("recv_local_offsets", []),
            dtype=torch.int32,
            device=device,
        ).contiguous()
        count = int(source.get("count", recv_local_offsets.numel()))
        if count != recv_local_offsets.numel():
            raise RuntimeError(f"recv count/local offset mismatch from src {src}: {count} vs {recv_local_offsets.numel()}")
        cpu_tensor = torch.empty((count, coord_dim), dtype=torch.float32)
        req = dist.irecv(cpu_tensor, src=src) if count > 0 else None
        # 字典中的value存的是引用
        pending.append({
            "req": req,
            "cpu_tensor": cpu_tensor,
            "recv_local_offsets": recv_local_offsets,
        })
    return pending


# step_entry是plan["send"][step]下的(["targets"][length]["dst/batch_indices"])
def launch_sends_for_step(step_entry, updated_coords, device):
    pending = []
    # 返回的是target对应的列表（长度为length）
    for target in normalize_entries(step_entry, "targets"):
        # dst_rank
        dst = int(target["dst"])
        # 通信vertex的编号
        batch_indices = torch.tensor(
            target.get("batch_indices", []),
            dtype=torch.long,
            device=device,
        )
        # 发送端payload
        cpu_payload = stage_send_payload(updated_coords, batch_indices)
        req = dist.isend(cpu_payload, dst=dst) if cpu_payload.numel() > 0 else None
        # TODO:cpu_payload没必要放在pending里
        pending.append({"req": req, "cpu_tensor": cpu_payload})
    return pending

# 把上一轮中所有“还挂着的异步发送”彻底等完，然后清空 pending_sends 列表
def finalize_pending_sends(pending_sends):
    for entry in pending_sends:
        req = entry["req"]
        if req is not None:
            req.wait()
    pending_sends.clear()


def infer_one_step(model, coord_dim, device):
    # 从ipc取出这一步要处理的子图数据
    batch = ipc_service.get_next(coord_dim)
    # 检查数据是否缺失
    if len(batch) < 5:
        return 0, None, None

    ids, features, labels, agg_src, agg_dst = batch
    block_src_num, block_dst_num = ipc_service.get_block_size()

    if block_dst_num == 0:
        return 0, None, None

    block = create_dgl_block(agg_src, agg_dst, block_src_num, block_dst_num)
    with torch.no_grad():
        # 应该是第 i 个目标节点的新坐标（i不是local_id）
        updated_coords = model(block, features)
        # 这里应该是目标节点在本地缓存里应该写的写入位置（编号）
        root_local_offsets = labels[:block_dst_num].contiguous()
        # 应该是把新坐标更新在本地
        ipc_service.update_coordinates(root_local_offsets, updated_coords.contiguous())

    torch.cuda.synchronize(device)
    return block_dst_num, root_local_offsets, updated_coords.contiguous()


def worker_main(args):
    # 推断确定当前 rank、GPU
    rank, local_rank, world_size = infer_rank_info()
    # 如果这是多进程/多节点运行，就初始化通信组
    init_dist_if_needed(rank, world_size)

    # 固定随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # CUDA 设备绑定
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 初始化 IPC 服务，并获取推理step与rank通信计划
    ipc_service.initialize()
    infer_steps, _, _ = ipc_service.get_steps()
    plan = load_rank_plan(args.partition_meta_path, rank, infer_steps, world_size > 1)

    # 创建模型并切到 eval 模式
    model = CoordinateUpdater(args.coord_dim, args.hidden_dim).to(device)
    model.eval()

    # 尚未完成的异步通信任务列表（通信异步）
    pending_recvs = []
    pending_sends = []

    for epoch in range(args.epoch):
        # 每个 epoch 的初始化
        start = time.time()
        updated_nodes = 0
        # 核心：step 循环
        for step in range(infer_steps):
            # 先应用上一步收到的远端更新
            if pending_recvs:
                apply_received_updates(device, pending_recvs)

            # 做一步本地推理，得到 updated_coords
            updated, _, updated_coords = infer_one_step(model, args.coord_dim, device)
            updated_nodes += updated

            # 如果是多卡分布式:
            if dist.is_initialized() and updated_coords is not None:
                # 确保上一轮发送已完成
                finalize_pending_sends(pending_sends)
                # 发送当前 step 的更新
                pending_sends = launch_sends_for_step(plan["send"][step], updated_coords, device)
                # 提前挂起下一 step 的接收
                pending_recvs = post_recvs_for_step(plan["recv"][step + 1], args.coord_dim, device)

            # 每一步结束时做一次 IPC 同步
            ipc_service.synchronize()

        # step 循环结束后，处理最后残留的通信（把最后残留的接收和发送收尾）
        if pending_recvs:
            apply_received_updates(device, pending_recvs)
        if pending_sends:
            finalize_pending_sends(pending_sends)
        # rank0 打印耗时和更新数
        if rank == 0:
            print(f"Epoch:{epoch}, Cost:{time.time() - start} s, Updated Roots: {updated_nodes}")

    ipc_service.finalize()
    finalize_dist_if_needed()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Coordinate inference.")
    argparser.add_argument("--coord_dim", type=int, default=128)
    argparser.add_argument("--hidden_dim", type=int, default=256)
    argparser.add_argument("--epoch", type=int, default=2)
    argparser.add_argument("--gpu_number", type=int, default=2)
    argparser.add_argument("--partition_meta_path", type=str, default="")
    argparser.add_argument("--seed", type=int, default=0)
    args = argparser.parse_args()

    worker_main(args)
