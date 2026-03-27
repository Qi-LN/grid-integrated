# Slurm Bring-Up

`run_original_train.sbatch` is a single-node Slurm launcher for the original Legion training flow.

Example:

```bash
sbatch --gres=gpu:2 \
  --export=ALL,DATASET_NAME=paper100m,GPU_NUMBER=2,EPOCH=2 \
  slurm/run_original_train.sbatch
```

Useful overrides:

- `MODEL=graphsage|gcn`
- `DATASET_NAME=products|paper100m`
- `DATASET_PATH=dataset`
- `GPU_NUMBER=2`
- `TRAIN_BATCH_SIZE=8000`
- `CLASS_NUM=172`
- `FEATURES_NUM=128`
- `HIDDEN_DIM=256`
- `CACHE_MEMORY=38000000`

The script keeps the original runtime order:

1. build `sampling_server`
2. build `training_backend/ipc_service` in place
3. start `legion_server.py`
4. wait for `System is ready for serving`
5. start the training backend
