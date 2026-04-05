#include <iostream>
#include <vector>
#include "helper_multiprocess.h"
#include <stdio.h>
#include <stdlib.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "ipc_service.h"

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#define MAX_DEVICE 8
#define MEMORY_USAGE 7

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

typedef struct shmStruct_st {
  int32_t steps[3];
  int32_t serve_mode;
  int32_t coord_shard_count;
  int64_t coord_shard_starts[MAX_DEVICE];
  int64_t coord_shard_sizes[MAX_DEVICE];
  cudaIpcMemHandle_t coordHandle[MAX_DEVICE];
  cudaIpcMemHandle_t memHandle[MAX_DEVICE][INTERBATCH_CON][MEMORY_USAGE];
} shmStruct;

class GPUIPCEnv : public IPCEnv {
public:
  int Initialize() override {
    volatile shmStruct *shm = NULL;
    int central_device = -1;
    cudaGetDevice(&central_device);
    cudaCheckError();
    central_device_ = central_device;
    sharedMemoryInfo info;
    const char shmName[] = "simpleIPCshm";
    if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
      printf("Failed to create shared memory slab\n");
      exit(EXIT_FAILURE);
    }

    shm = (volatile shmStruct *)info.addr;
    train_step_ = shm->steps[0];
    valid_step_ = shm->steps[1];
    test_step_ = shm->steps[2];
    ids_.resize(INTERBATCH_CON);
    float_features_.resize(INTERBATCH_CON);
    labels_.resize(INTERBATCH_CON);
    agg_src_.resize(INTERBATCH_CON);
    agg_dst_.resize(INTERBATCH_CON);
    node_counter_.resize(INTERBATCH_CON);
    edge_counter_.resize(INTERBATCH_CON);
    coord_shard_count_ = shm->coord_shard_count;
    local_coordinate_shard_ = nullptr;

    for(int i = 0; i < INTERBATCH_CON; i++){
      cudaIpcOpenMemHandle(&ids_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][0], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&float_features_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][1], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&labels_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][2], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&agg_src_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][3], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&agg_dst_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][4], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&node_counter_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][5], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&edge_counter_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][6], cudaIpcMemLazyEnablePeerAccess);
      cudaCheckError();
    }

    if (coord_shard_count_ > 0) {
      // 训练端装载shard指针的容器
      coordinate_shards_.resize(coord_shard_count_, nullptr);
      for (int32_t shard = 0; shard < coord_shard_count_; ++shard) {
        // 从共享内存 shm->coordHandle[shard] 里取出一个 CUDA IPC handle，然后在当前进程中把它打开，得到一个可用的设备指针，存到 coordinate_shards_[shard] 里。
        // if (shm->coord_shard_sizes[shard] > 0) {
        //   cudaIpcOpenMemHandle(&coordinate_shards_[shard], *(cudaIpcMemHandle_t*)&shm->coordHandle[shard], cudaIpcMemLazyEnablePeerAccess);
        //   cudaCheckError();
        // }
        if (shm->coord_shard_sizes[shard] > 0) {
          void* shard_ptr = nullptr;
          cudaIpcOpenMemHandle(&shard_ptr,
                              *(cudaIpcMemHandle_t*)&shm->coordHandle[shard],
                              cudaIpcMemLazyEnablePeerAccess);
          cudaCheckError();
          coordinate_shards_[shard] = static_cast<float*>(shard_ptr);
        }
      }
      // 单独保存一份本卡的shard指针
      // 更新坐标时，当前训练进程只需要写自己这一张卡对应的本地 shard
      if (central_device_ >= 0 && central_device_ < coord_shard_count_) {
        local_coordinate_shard_ = coordinate_shards_[central_device_];
      }
    }
    std::cout<<"CUDA: "<<central_device<<" IPC shared memory opened\n";

    semr_.resize(INTERBATCH_CON);
    semw_.resize(INTERBATCH_CON);
    for(int i = 0; i < INTERBATCH_CON; i++){
      std::string ssr = "sem_r_";
      std::string ssw = "sem_w_";
      std::string ssri = ssr + std::to_string(central_device) + "_" + std::to_string(i);
      std::string sswi = ssw + std::to_string(central_device) + "_" + std::to_string(i);
      semr_[i] = sem_open(ssri.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semr_[i] == SEM_FAILED ){
        printf("errno = %d\n", errno );
        return -1;
      }
      semw_[i] = sem_open(sswi.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semw_[i] == SEM_FAILED){
        printf("errno = %d\n", errno );
        return -1;
      }
      sem_post(semr_[i]);
    }

    current_pipe_ = 0;
    return central_device;
  }

  void Wait() override {
    sem_t* sem = semw_[current_pipe_];
    sem_wait(sem);
  }

  void Post() override {
    sem_t* sem = semr_[current_pipe_];
    sem_post(sem);
    current_pipe_ = (current_pipe_ + 1)%INTERBATCH_CON;
  }

  int32_t* GetIds() override {
    return (int32_t*)ids_[current_pipe_];
  }
  float* GetFloatFeatures() override {
    return (float*)float_features_[current_pipe_];
  }
  int32_t* GetLabels() override {
    return (int32_t*)labels_[current_pipe_];
  }
  int32_t* GetAggSrc() override {
    return (int32_t*)agg_src_[current_pipe_];
  }
  int32_t* GetAggDst() override {
    return (int32_t*)agg_dst_[current_pipe_];
  }
  int32_t* GetNodeCounter() override {
    return (int32_t*)(node_counter_[current_pipe_]);
  }
  int32_t* GetEdgeCounter() override {
    return (int32_t*)(edge_counter_[current_pipe_]);
  }
  float* GetLocalCoordinateShard() override {
    return local_coordinate_shard_;
  }
  int32_t GetTrainStep() override {
    return train_step_;
  }
  int32_t GetValidStep() override {
    return valid_step_;
  }
  int32_t GetTestStep() override {
    return test_step_;
  }

  void Finalize() override {
    for(int i = 0; i < INTERBATCH_CON; i++){
      cudaIpcCloseMemHandle(ids_[i]);
      cudaIpcCloseMemHandle(float_features_[i]);
      cudaIpcCloseMemHandle(labels_[i]);
      cudaIpcCloseMemHandle(agg_src_[i]);
      cudaIpcCloseMemHandle(agg_dst_[i]);
      cudaIpcCloseMemHandle(node_counter_[i]);
      cudaIpcCloseMemHandle(edge_counter_[i]);
      sem_t* sem = semw_[i];
      if(sem_close(sem) == -1){
        std::cout<<"close sem "<<i<<" failed\n";
      }
    }
    // 关闭handle
    for (int32_t shard = 0; shard < coord_shard_count_; ++shard) {
      if (coordinate_shards_[shard] != nullptr) {
        cudaIpcCloseMemHandle(coordinate_shards_[shard]);
      }
    }
  }
private:
  std::vector<void*> ids_;
  std::vector<void*> float_features_;
  std::vector<void*> labels_;
  std::vector<void*> agg_src_;
  std::vector<void*> agg_dst_;
  std::vector<void*> node_counter_;
  std::vector<void*> edge_counter_;
  std::vector<float*> coordinate_shards_;
  std::vector<sem_t*> semw_;
  std::vector<sem_t*> semr_;
  // 特征更新时本地修改的指针
  float* local_coordinate_shard_;

  int32_t train_step_;
  int32_t valid_step_;
  int32_t test_step_;
  int32_t coord_shard_count_;
  int32_t central_device_;
  int current_pipe_;
};

IPCEnv* NewIPCEnv(){
  return new GPUIPCEnv();
}

// 把当前 batch 里推理得到的 updated_feat，按 root_local_offsets 写回本地 GPU 的 coord shard。
__global__ void update_local_coordinate_shard_kernel(
    int32_t* root_local_offsets,
    // 推理后算出来的新特征[root_count, coord_dim]
    float* updated_coords,
    int32_t root_count,
    int32_t coord_dim,
    float* local_coordinate_shard) {
  for (int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
       thread_idx < int64_t(root_count) * coord_dim;
       thread_idx += blockDim.x * gridDim.x) {
    // 当前是 batch 里的第几个 root
    int32_t root_offset = thread_idx / coord_dim;
    // 这个 root 的第几个坐标/特征维度
    int32_t feat_offset = thread_idx % coord_dim;
    // root_local_offsets:batch中的第 i 个 root节点 -> 它在当前本地 shard 中的 local offset
    int32_t local_offset = root_local_offsets[root_offset];
    if (local_offset >= 0) {
      // 新节点特征更新在shard中
      local_coordinate_shard[int64_t(local_offset) * coord_dim + feat_offset] =
          updated_coords[int64_t(root_offset) * coord_dim + feat_offset];
    }
  }
}

std::vector<torch::Tensor> cuda_get_next(
    int32_t* ids,
    float* float_features,
    int32_t* labels,
    int feature_dim,
    int32_t* agg_src,
    int32_t* agg_dst,
    int32_t* node_counter,
    int32_t* edge_counter,
    int32_t* h_node_counter,
    int32_t* h_edge_counter
    ){
    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);
    cudaCheckError();

    cudaMemcpy(h_node_counter, node_counter, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge_counter, edge_counter, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    int hop_num = h_node_counter[INTRABATCH_CON * 3 - 1];

    std::vector<torch::Tensor> ret;

    torch::Tensor ids_tensor = torch::from_blob(
      ids,
      {(long long)h_node_counter[INTRABATCH_CON * 3 + hop_num]},
      torch::TensorOptions().dtype(torch::kI32).device(device));
    ret.push_back(ids_tensor);

    torch::Tensor feature_tensor = torch::from_blob(
      float_features,
      {(long long)(h_node_counter[INTRABATCH_CON * 3 + hop_num]), (long long)(feature_dim)},
      torch::TensorOptions().dtype(torch::kF32).device(device));
    ret.push_back(feature_tensor);

    torch::Tensor labels_tensor = torch::from_blob(
      labels,
      {(long long)h_node_counter[INTRABATCH_CON * 3]},
      torch::TensorOptions().dtype(torch::kI32).device(device));
    ret.push_back(labels_tensor);

    for(int i = hop_num; i > 0; i--){
      torch::Tensor agg_src_tensor = torch::from_blob(
        agg_src,
          {(long long)h_edge_counter[INTRABATCH_CON * 3 + i]},
          torch::TensorOptions().dtype(torch::kI32).device(device));
      torch::Tensor agg_dst_tensor = torch::from_blob(
        agg_dst,
          {(long long)h_edge_counter[INTRABATCH_CON * 3 + i]},
          torch::TensorOptions().dtype(torch::kI32).device(device));
      ret.push_back(agg_src_tensor);
      ret.push_back(agg_dst_tensor);
    }

    return ret;
}

void cuda_update_coordinates(
    int32_t* root_local_offsets,
    float* updated_coords,
    int32_t root_count,
    int32_t coord_dim,
    float* local_coordinate_shard) {
    if (root_count <= 0 || coord_dim <= 0 || local_coordinate_shard == nullptr) {
      return;
    }
    dim3 block_num(32, 1);
    dim3 thread_num(256, 1);
    update_local_coordinate_shard_kernel<<<block_num, thread_num>>>(
        root_local_offsets,
        updated_coords,
        root_count,
        coord_dim,
        local_coordinate_shard);
    cudaCheckError();
}
