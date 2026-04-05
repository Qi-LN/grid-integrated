import argparse
import time

import dgl
import ipc_service
import torch
import torch.multiprocessing as mp
from dgl.heterograph import DGLBlock
from dgl.nn.pytorch import SAGEConv

# 把 IPC 返回的 agg_src / agg_dst 和节点数，组装成一个 DGL 的 DGLBlock
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

# 定义轻量模型
class CoordinateUpdater(torch.nn.Module):
    def __init__(self, coord_dim, hidden_dim):
        super().__init__()
        self.conv = SAGEConv(coord_dim, hidden_dim, "mean")
        self.proj = torch.nn.Linear(hidden_dim, coord_dim)

    def forward(self, block, features):
        hidden = self.conv(block, features)
        return self.proj(hidden)


def infer_one_step(model, coord_dim, device):
    batch = ipc_service.get_next(coord_dim)
    if len(batch) < 5:
        ipc_service.synchronize()
        return 0

    ids, features, labels, agg_src, agg_dst = batch
    block_src_num, block_dst_num = ipc_service.get_block_size()

    if block_dst_num == 0:
        ipc_service.synchronize()
        return 0

    block = create_dgl_block(agg_src, agg_dst, block_src_num, block_dst_num)
    with torch.no_grad():
        # 前向推理
        updated_coords = model(block, features)
        print("推理成功")
        # labels 在 infer 模式下不是分类标签，而是“这些 root 在本地 shard 中的位置偏移”
        root_local_offsets = labels[:block_dst_num].contiguous()
        # 把 labels[:block_dst_num] 作为 root_local_offsets 传给 IPC 服务，告诉它哪些节点的坐标被更新了
        ipc_service.update_coordinates(root_local_offsets, updated_coords.contiguous())
        print("坐标更新成功")

    torch.cuda.synchronize(device)
    ipc_service.synchronize()
    return block_dst_num


# 用多进程按 GPU 数启动
def worker_process(rank, world_size, args):
    print(f"Running coordinate inference on CUDA {rank}.")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    ipc_service.initialize()
    infer_steps, _, _ = ipc_service.get_steps()

    model = CoordinateUpdater(args.coord_dim, args.hidden_dim).to(device)
    model.eval()

    for epoch in range(args.epoch):
        start = time.time()
        updated_nodes = 0
        for step in range(infer_steps):
            updated_nodes += infer_one_step(model, args.coord_dim, device)
        if rank == 0:
            print(
                "Epoch:{}, Cost:{} s, Updated Roots: {}".format(
                    epoch, time.time() - start, updated_nodes
                )
            )

    ipc_service.finalize()


def run_distribute(dist_fn, world_size, args):
    mp.spawn(dist_fn, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Coordinate inference.")
    argparser.add_argument("--coord_dim", type=int, default=128)
    argparser.add_argument("--hidden_dim", type=int, default=256)
    argparser.add_argument("--epoch", type=int, default=2)
    argparser.add_argument("--gpu_number", type=int, default=2)
    args = argparser.parse_args()

    run_distribute(worker_process, args.gpu_number, args)
