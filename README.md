
## Multi-Node Coordinate Inference Metadata

For the full multi-node test procedure, see:

```bash
slurm/MULTI_NODE_COORDINATE_INFER.md
```

`run_coordinate_infer.sbatch` expects halo metadata under:

```bash
$DATASET_PATH/$DATASET_NAME/halo_partition
```

Generate it with DGL/METIS partitioning and one-hop halo expansion:

```bash
python dataset/gen_halo_partition_meta.py \
  --dataset_path dataset \
  --dataset_name products \
  --num_nodes 2 \
  --gpus_per_node 8 \
  --batch_size 800
```

The output contains:
- `infer_steps.txt`
- `node_<node_rank>/gpu_<gpu_id>_owner_roots.bin`
- `node_<node_rank>/gpu_<gpu_id>_halo_nodes.bin`
- `rank_<global_rank>.json`
- `summary.json`

For a one-shot run that regenerates metadata before launch:

```bash
GENERATE_HALO_META=1 sbatch slurm/run_coordinate_infer.sbatch
```
