#include <fcntl.h>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <torch/extension.h>
#include "ipc_service.h"


IPCEnv* env;
int32_t h_node_counter[16];
int32_t h_edge_counter[16];

void InitializeIPC(){
    env = NewIPCEnv();
    env->Initialize();
}

void FinalizeIPC(){
    env->Finalize();
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
    );

void cuda_update_coordinates(
    int32_t* root_local_offsets,
    float* updated_coords,
    int32_t root_count,
    int32_t coord_dim,
    float* local_coordinate_shard);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> get_next(int feature_dim) {
    env->Wait();
    int32_t* ids = env->GetIds();
    float* float_features = env->GetFloatFeatures();
    int32_t* labels = env->GetLabels();
    int32_t* agg_src = env->GetAggSrc();
    int32_t* agg_dst = env->GetAggDst();
    int32_t* node_counter = env->GetNodeCounter();
    int32_t* edge_counter = env->GetEdgeCounter();
    auto result = cuda_get_next(ids, float_features, labels,
                                feature_dim,
                                agg_src, agg_dst, 
                                node_counter, edge_counter, 
                                h_node_counter, h_edge_counter);
    return result;
}

std::vector<int> get_block_size() {
    std::vector<int> ret;
    int hop_num = h_node_counter[INTRABATCH_CON * 3 - 1];

    for(int i = hop_num; i > 0; i--){
        ret.push_back(h_node_counter[INTRABATCH_CON * 3 + i]);
        ret.push_back(h_node_counter[INTRABATCH_CON * 3 + i - 1]);
    }
    return ret;
}

std::vector<int32_t> get_steps(){
    std::vector<int32_t> ret;
    ret.push_back(env->GetTrainStep());
    ret.push_back(env->GetValidStep());
    ret.push_back(env->GetTestStep());
    return ret;
} 

void Synchronize(){
    env->Post();
}

void update_coordinates(torch::Tensor root_local_offsets, torch::Tensor updated_coords) {
    CHECK_INPUT(root_local_offsets);
    CHECK_INPUT(updated_coords);
    float* local_coordinate_shard = env->GetLocalCoordinateShard();
    if (local_coordinate_shard == nullptr) {
        return;
    }
    int32_t root_count = root_local_offsets.size(0);
    int32_t coord_dim = updated_coords.size(1);
    cuda_update_coordinates(
        root_local_offsets.data_ptr<int32_t>(),
        updated_coords.data_ptr<float>(),
        root_count,
        coord_dim,
        local_coordinate_shard);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_next", &get_next, "dataset get next (CUDA)");
  m.def("get_block_size", &get_block_size, "get dgl block size(CUDA)");
  m.def("get_steps", &get_steps, "get steps(CUDA)");
  m.def("initialize", &InitializeIPC, "InitializeIPC (CUDA)");
  m.def("finalize", &FinalizeIPC, "FinalizeIPC (CUDA)");
  m.def("synchronize", &Synchronize, "synchronize (CUDA)");
  m.def("update_coordinates", &update_coordinates, "update_coordinates (CUDA)");
}
