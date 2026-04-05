#!/bin/bash
#SBATCH --job-name=legion-infer
#SBATCH -p Nvidia_A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --output=infer.out

set -euo pipefail

#REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
#cd "$REPO_ROOT"
REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR:-$PWD}/.." && pwd)"
cd "$REPO_ROOT"


# 加载环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /public/home/liufang395/anaconda3/envs/megatron_env
module load compiler/gcc/11.5.0

# 环境变量，分布式Master_addr和port
export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((12000 + (${SLURM_JOB_ID:-1} % 20000)))}"

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
hash -r

SERVER_LOG="${SERVER_LOG:-$REPO_ROOT/slurm/server.log}"

DATASET_NAME="${DATASET_NAME:-products}"
DATASET_PATH="${DATASET_PATH:-dataset}"
ROOTSET_PATH="${ROOTSET_PATH:-$DATASET_PATH/$DATASET_NAME/rootset}"
EPOCH="${EPOCH:-2}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
# HOPS_NUM="${HOPS_NUM:-1}"
GPU_NUMBER="${GPU_NUMBER:-8}"

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

make -C sampling_server clean
make -C sampling_server -j "$BUILD_JOBS"

(
    cd training_backend
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}" python setup.py install  #build_ext --inplace
)

# python legion_server.py \
stdbuf -oL -eL python legion_server.py \
    --dataset_path "$DATASET_PATH" \
    --dataset_name "$DATASET_NAME" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --gpu_number "$GPU_NUMBER" \
    --epoch "$EPOCH" \
    --cache_memory 0 \
    --mode infer \
    --rootset_path "$ROOTSET_PATH" \
    --usenvlink 0 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

SECONDS_WAITED=0
until grep -q "ready for serving" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "Inference server exited before becoming ready."
        cat "$SERVER_LOG"
        exit 1
    fi
    if [ "$SECONDS_WAITED" -ge "$SERVER_READY_TIMEOUT" ]; then
        echo "Timed out waiting for inference server readiness."
        tail -n 200 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
    SECONDS_WAITED=$((SECONDS_WAITED + 5))
done

python training_backend/legion_coordinate_infer.py \
    --coord_dim "$COORD_DIM" \
    --hidden_dim "$HIDDEN_DIM" \
    --gpu_number "$GPU_NUMBER" \
    --epoch "$EPOCH"

wait "$SERVER_PID"
