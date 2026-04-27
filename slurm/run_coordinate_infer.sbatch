#!/bin/bash
#SBATCH --job-name=legion-infer
#SBATCH -p gpu_a800
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --output=slurm/infer.out

set -euo pipefail

REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$PWD}/.." && pwd)"
cd "$REPO_ROOT"

# 魏桥加载环境
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate /public/home/liufang395/anaconda3/envs/megatron_env
# module load compiler/gcc/11.5.0

# 并行加载环境
conda init >/dev/null 2>&1 || true
conda activate legion
module load cuda/12.4

export PYTHONUNBUFFERED=1
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-$((12000 + (${SLURM_JOB_ID:-1} % 20000)))}"
export MASTER_ADDR MASTER_PORT

SERVER_LOG_DIR="${SERVER_LOG_DIR:-$REPO_ROOT/slurm}"
mkdir -p "$SERVER_LOG_DIR"

DATASET_NAME="${DATASET_NAME:-products}"
DATASET_PATH="${DATASET_PATH:-dataset}"
ROOTSET_PATH="${ROOTSET_PATH:-$DATASET_PATH/$DATASET_NAME/rootset}"
PARTITION_META_PATH="${PARTITION_META_PATH:-$DATASET_PATH/$DATASET_NAME/halo_partition}"
EPOCH="${EPOCH:-2}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
GPU_NUMBER="${GPU_NUMBER:-8}"
GENERATE_HALO_META="${GENERATE_HALO_META:-0}"

case "$DATASET_NAME" in
    products)
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-800}"
        COORD_DIM="${COORD_DIM:-100}"
        HIDDEN_DIM="${HIDDEN_DIM:-128}"
        ;;
    paper100m)
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8000}"
        COORD_DIM="${COORD_DIM:-128}"
        HIDDEN_DIM="${HIDDEN_DIM:-256}"
        ;;
    *)
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8000}"
        COORD_DIM="${COORD_DIM:-128}"
        HIDDEN_DIM="${HIDDEN_DIM:-256}"
        ;;
esac

if [ "$GENERATE_HALO_META" = "1" ]; then
    python dataset/gen_halo_partition_meta.py \
        --dataset_path "$DATASET_PATH" \
        --dataset_name "$DATASET_NAME" \
        --num_nodes "$SLURM_JOB_NUM_NODES" \
        --gpus_per_node "$GPU_NUMBER" \
        --batch_size "$TRAIN_BATCH_SIZE" \
        --output_path "$PARTITION_META_PATH"
fi

make -C sampling_server clean
make -C sampling_server -j "$BUILD_JOBS"
(
    cd training_backend
    python setup.py install
)

srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -lc '
set -euo pipefail
cd '"$REPO_ROOT"'
export PYTHONUNBUFFERED=1
export MASTER_ADDR='"$MASTER_ADDR"'
export MASTER_PORT='"$MASTER_PORT"'
export LEGION_IPC_NAMESPACE="'"${SLURM_JOB_ID:-manual}"'_${SLURM_NODEID}"
LOG_FILE="'"$SERVER_LOG_DIR"'/server-${SLURM_JOB_ID:-manual}-${SLURM_NODEID}.log"
stdbuf -oL -eL python legion_server.py     --dataset_path '"$DATASET_PATH"'     --dataset_name '"$DATASET_NAME"'     --train_batch_size '"$TRAIN_BATCH_SIZE"'     --gpu_number '"$GPU_NUMBER"'     --local_gpu_number '"$GPU_NUMBER"'     --num_nodes '"$SLURM_JOB_NUM_NODES"'     --node_rank ${SLURM_NODEID}     --epoch '"$EPOCH"'     --cache_memory 0     --mode infer     --rootset_path '"$ROOTSET_PATH"'     --partition_meta_path '"$PARTITION_META_PATH"'     --usenvlink 0     > "$LOG_FILE" 2>&1
' &
SERVER_STEP_PID=$!

cleanup() {
    if kill -0 "$SERVER_STEP_PID" >/dev/null 2>&1; then
        kill "$SERVER_STEP_PID" >/dev/null 2>&1 || true
        wait "$SERVER_STEP_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

SECONDS_WAITED=0
while true; do
    READY_COUNT=$(grep -l "ready for serving" "$SERVER_LOG_DIR"/server-${SLURM_JOB_ID:-manual}-*.log 2>/dev/null | wc -l || true)
    if [ "$READY_COUNT" -ge "$SLURM_JOB_NUM_NODES" ]; then
        break
    fi
    if ! kill -0 "$SERVER_STEP_PID" >/dev/null 2>&1; then
        echo "At least one inference server exited before becoming ready."
        tail -n 200 "$SERVER_LOG_DIR"/server-${SLURM_JOB_ID:-manual}-*.log || true
        exit 1
    fi
    if [ "$SECONDS_WAITED" -ge "$SERVER_READY_TIMEOUT" ]; then
        echo "Timed out waiting for inference servers readiness."
        tail -n 200 "$SERVER_LOG_DIR"/server-${SLURM_JOB_ID:-manual}-*.log || true
        exit 1
    fi
    sleep 5
    SECONDS_WAITED=$((SECONDS_WAITED + 5))
done

srun --overlap --kill-on-bad-exit=1 --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$((SLURM_JOB_NUM_NODES * GPU_NUMBER))" --ntasks-per-node="$GPU_NUMBER" bash -lc '
set -euo pipefail
cd '"$REPO_ROOT"'
export PYTHONUNBUFFERED=1
export MASTER_ADDR='"$MASTER_ADDR"'
export MASTER_PORT='"$MASTER_PORT"'
export LEGION_IPC_NAMESPACE="'"${SLURM_JOB_ID:-manual}"'_${SLURM_NODEID}"
python training_backend/legion_coordinate_infer.py     --coord_dim '"$COORD_DIM"'     --hidden_dim '"$HIDDEN_DIM"'     --gpu_number '"$GPU_NUMBER"'     --epoch '"$EPOCH"'     --partition_meta_path '"$PARTITION_META_PATH"'
'

wait "$SERVER_STEP_PID"
