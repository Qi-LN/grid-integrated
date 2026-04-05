#!/bin/bash
#SBATCH --job-name=legion-train
#SBATCH --partition=Nvidia_A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --output=legion-train-%j.out

set -euo pipefail

# 确定仓库根目录并切换过去
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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


MODEL="${MODEL:-graphsage}"
DATASET_NAME="${DATASET_NAME:-products}"
DATASET_PATH="${DATASET_PATH:-dataset}"
EPOCH="${EPOCH:-2}"
CACHE_MEMORY="${CACHE_MEMORY:-38000000}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
SERVER_LOG="${SERVER_LOG:-$REPO_ROOT/server-${SLURM_JOB_ID:-manual}.log}"

GPU_NUMBER="${GPU_NUMBER:-8}"

case "$DATASET_NAME" in
    products)
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-800}"
        CLASS_NUM="${CLASS_NUM:-47}"
        FEATURES_NUM="${FEATURES_NUM:-100}"
        HIDDEN_DIM="${HIDDEN_DIM:-128}"
        ;;
    paper100m)
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8000}"
        CLASS_NUM="${CLASS_NUM:-172}"
        FEATURES_NUM="${FEATURES_NUM:-128}"
        HIDDEN_DIM="${HIDDEN_DIM:-256}"
        ;;
    *)
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8000}"
        CLASS_NUM="${CLASS_NUM:-2}"
        FEATURES_NUM="${FEATURES_NUM:-128}"
        HIDDEN_DIM="${HIDDEN_DIM:-256}"
        ;;
esac

HOPS_NUM="${HOPS_NUM:-2}"

case "$MODEL" in
    graphsage)
        TRAINING_SCRIPT="training_backend/legion_graphsage.py"
        ;;
    gcn)
        TRAINING_SCRIPT="training_backend/legion_gcn.py"
        ;;
    *)
        echo "Unsupported MODEL=$MODEL. Use MODEL=graphsage or MODEL=gcn."
        exit 1
        ;;
esac

echo "Repo root:        $REPO_ROOT"
echo "Dataset:          $DATASET_NAME"
echo "Training script:  $TRAINING_SCRIPT"
echo "GPU number:       $GPU_NUMBER"
echo "Master addr/port: $MASTER_ADDR:$MASTER_PORT"
echo "Server log:       $SERVER_LOG"

# 手动编译
# make -C sampling_server clean
# make -C sampling_server -j "$BUILD_JOBS"

# (
#     cd training_backend
#     CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" python setup.py build_ext --inplace
# )

python legion_server.py \
    --dataset_path "$DATASET_PATH" \
    --dataset_name "$DATASET_NAME" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --fanout "[25,10]" \
    --gpu_number "$GPU_NUMBER" \
    --epoch "$EPOCH" \
    --cache_memory "$CACHE_MEMORY" \
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
until grep -q "System is ready for serving" "$SERVER_LOG" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "Sampling server exited before becoming ready."
        cat "$SERVER_LOG"
        exit 1
    fi
    if [ "$SECONDS_WAITED" -ge "$SERVER_READY_TIMEOUT" ]; then
        echo "Timed out waiting for sampling server readiness."
        tail -n 200 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 5
    SECONDS_WAITED=$((SECONDS_WAITED + 5))
done

python "$TRAINING_SCRIPT" \
    --class_num "$CLASS_NUM" \
    --features_num "$FEATURES_NUM" \
    --hidden_dim "$HIDDEN_DIM" \
    --hops_num "$HOPS_NUM" \
    --gpu_number "$GPU_NUMBER" \
    --epoch "$EPOCH"

wait "$SERVER_PID"
