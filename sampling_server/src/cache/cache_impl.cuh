#pragma once
#ifndef CACHE_IMPL_H
#define CACHE_IMPL_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <mutex>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <functional>
#include <cstdlib>

#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#define MIN_INTERVAL 0.01
#define CLS 64

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <bcht.hpp>
#include <cmd.hpp>
#include <gpu_timer.hpp>
#include <limits>
#include <perf_report.hpp>
#include <rkg.hpp>
#include <type_traits>

#include "system_config.cuh"

using pair_type = bght::pair<int32_t, int32_t>;
using index_pair_type = bght::pair<int32_t, char>;
using offset_pair_type = bght::pair<int32_t, int32_t>;

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }



__global__ void GetEdgeMem(int32_t* edge_order, uint64_t* edge_mem, int32_t total_num_nodes, int64_t* csr_index){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < total_num_nodes; thread_idx += gridDim.x * blockDim.x){
        int32_t id = edge_order[thread_idx];
        int64_t neighbor_count = csr_index[id + 1]- csr_index[id];
        edge_mem[thread_idx] = (sizeof(int64_t) + sizeof(int32_t) * neighbor_count);
    }
}


__global__ void aggregate_access(unsigned long long int* agg_access_time, unsigned long long int* new_access_time, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (total_num_nodes); thread_idx += blockDim.x * gridDim.x){
        agg_access_time[thread_idx] += new_access_time[thread_idx];
    }
}


__global__ void init_cache_order(int32_t* cache_order, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (total_num_nodes); thread_idx += blockDim.x * gridDim.x){
        cache_order[thread_idx] = thread_idx;
    }
}

__global__ void init_feature_cache(float** pptr, float* ptr, int dev_id){
    pptr[dev_id] = ptr;
}

__global__ void InitIndexPair(index_pair_type* pair, int32_t* QT, int32_t capacity, int32_t cache_expand, int32_t Kg, int32_t Ki){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity * cache_expand; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = QT[thread_idx];
        pair[thread_idx].second = thread_idx % Kg + Ki * Kg;
    }
}

__global__ void InitOffsetPair(offset_pair_type* pair, int32_t* QT, int32_t capacity, int32_t cache_expand, int32_t Kg){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity * cache_expand; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = QT[thread_idx];
        pair[thread_idx].second = thread_idx / Kg;
    }
}

//interleave cache
__global__ void InitPair(pair_type* pair, int32_t* QF, int32_t capacity, int32_t cache_expand, int32_t Kg){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity * cache_expand; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = QF[thread_idx];
        pair[thread_idx].second = (thread_idx % Kg) * capacity + thread_idx / Kg;
    }
}


//cpu cache & gpu cache
__global__ void HybridInitPair(pair_type* pair, int32_t* QF, int32_t cpu_cache_capacity, int32_t gpu_cache_capacity){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (cpu_cache_capacity + gpu_cache_capacity); thread_idx += gridDim.x * blockDim.x){
        if(thread_idx < gpu_cache_capacity){
            pair[thread_idx].first = QF[thread_idx];
            pair[thread_idx].second = cpu_cache_capacity + thread_idx;
        }else{
            pair[thread_idx].first = QF[thread_idx];
            pair[thread_idx].second = thread_idx - gpu_cache_capacity;
        }
    }
}


__global__ void topo_cache_hit(char* partition_index, int32_t batch_size, int32_t* global_count){
    __shared__ int32_t local_count[1];
    if(threadIdx.x == 0){
        local_count[0] = 0;
    }
    __syncthreads();
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (batch_size); thread_idx += blockDim.x * gridDim.x){
        int32_t offset = partition_index[thread_idx];
        if(offset >= 0){
            atomicAdd(local_count, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(global_count, local_count[0]);
    }
}


__global__ void feature_cache_hit(int32_t* cache_offset, int32_t batch_size, int32_t* global_count){
    __shared__ int32_t local_count[1];
    if(threadIdx.x == 0){
        local_count[0] = 0;
    }
    __syncthreads();
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (batch_size); thread_idx += blockDim.x * gridDim.x){
        int32_t offset = cache_offset[thread_idx];
        if(offset >= 0){
            atomicAdd(local_count, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(global_count, local_count[0]);
    }
}

__global__ void CacheHitTimes(int32_t* sampled_node, int32_t* node_counter){
    __shared__ int32_t count[2];
    if(threadIdx.x == 0){
        count[0] = 0;
        count[1] = 0;
    }
    __syncthreads();
    int32_t num_nodes = node_counter[INTRABATCH_CON * 2 + 1];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_nodes; thread_idx += gridDim.x * blockDim.x){
        int32_t cid = sampled_node[thread_idx];
        if(cid >= 0){
            atomicAdd(count, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(node_counter + INTRABATCH_CON * 2 + 2, count[0]);
    }
}

__global__ void FeatFillUp(int32_t capacity, int32_t float_feature_len, float* feature_cache, float* cpu_float_feature, int32_t* QF, int32_t Kg, int32_t Ki){
    for(int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < int64_t(capacity) * float_feature_len; thread_idx += gridDim.x * blockDim.x){
        int32_t id = QF[(thread_idx / float_feature_len) * Kg + Ki];
        feature_cache[thread_idx] = cpu_float_feature[int64_t(id) * float_feature_len + thread_idx % float_feature_len];
    }
}

__global__ void HotnessMeasure(int32_t* new_batch_ids, int32_t* node_counter, unsigned long long int* access_map){
    int32_t num_candidates = node_counter[INTRABATCH_CON * 2 + 1];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_candidates; thread_idx += gridDim.x * blockDim.x){
        int32_t cid = new_batch_ids[thread_idx];
        if(cid >= 0){
            atomicAdd(access_map + cid, 1);
        }
    }
}



__global__ void feat_cache_lookup(
	float* cpu_float_feature, float* gpu_float_feature, int32_t float_feature_len,
	int32_t* sampled_ids, int32_t* cache_index, 
    int32_t cpu_cache_capacity, int32_t gpu_cache_capacity,
	int32_t* node_counter, float* dst_float_buffer,
	int32_t op_id)
{
    int32_t node_off = 0;
	int32_t batch_size = 0;

    node_off   = node_counter[(op_id % INTRABATCH_CON) * 2];
    batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("%d %d\n", node_off, batch_size);
    // }
	if(float_feature_len > 0){
		for(int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (int64_t(batch_size) * float_feature_len); thread_idx += blockDim.x * gridDim.x){
			int32_t gidx;//global cache index
            int32_t fidx;//local cache index
            int32_t didx;//device index
            int32_t foffset;
            gidx = (cache_index[thread_idx / float_feature_len]);
			foffset = thread_idx % float_feature_len;
			if(gidx < cpu_cache_capacity && gidx >= 0){/*cache in cpu*/
				fidx = gidx % cpu_cache_capacity;//sampled_ids[node_off + (thread_idx / float_feature_len)];
				dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_feature_len)) + thread_idx)] = cpu_float_feature[int64_t(int64_t(int64_t(fidx) * float_feature_len) + foffset)];
			}else if(gidx >= cpu_cache_capacity){/*cache in gpu*/
                // didx = (gidx - cpu_cache_capacity) / gpu_cache_capacity;//device idx in clique
                fidx = (gidx - cpu_cache_capacity) % gpu_cache_capacity;
				dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_feature_len)) + thread_idx)] = gpu_float_feature[int64_t(int64_t(int64_t(fidx) * float_feature_len) + foffset)];
			}
		}
	}
}



__global__ void multiGPU_feat_cache_lookup(
	float* cpu_float_features, float** gpu_float_feature, int32_t float_feature_len,
	int32_t* sampled_ids, int32_t* cache_index, int32_t cache_capacity,
	int32_t* node_counter, float* dst_float_buffer,
	int32_t total_num_nodes,
	int32_t dev_id,
	int32_t op_id)
{
    int32_t node_off = 0;
	int32_t batch_size = 0;

    node_off   = node_counter[(op_id % INTRABATCH_CON) * 2];
    batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
	int32_t gidx;//global cache index
	int32_t fidx;//local cache index
	int32_t didx;//device index
	int32_t foffset;
	if(float_feature_len > 0){
		for(int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (int64_t(batch_size) * float_feature_len); thread_idx += blockDim.x * gridDim.x){
			gidx = (cache_index[thread_idx / float_feature_len]);
			didx = gidx / cache_capacity;//device idx in clique
			fidx = gidx % cache_capacity;
			foffset = thread_idx % float_feature_len;
			if(gidx < 0){/*cache miss*/
				fidx = sampled_ids[node_off + (thread_idx / float_feature_len)];
				if(fidx >= 0){
					dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_feature_len)) + thread_idx)] = cpu_float_features[int64_t(int64_t(int64_t(fidx%total_num_nodes) * float_feature_len) + foffset)];
				}
			}else{/*cache hit, find global position*/
				dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_feature_len)) + thread_idx)] = gpu_float_feature[didx][int64_t(int64_t(int64_t(fidx) * float_feature_len) + foffset)];
			}
		}
	}
}

// 按照 shard_ids 给出的节点列表，从整张图的特征矩阵 cpu_float_features 里把对应节点的特征抽出来，紧凑地拷到当前 shard 的连续存储 coordinate_shard 中。
__global__ void gather_coordinate_shard(
    float* cpu_float_features,
    int32_t float_feature_len,
    int32_t* shard_ids,
    int64_t shard_size,
    float* coordinate_shard)
{
    for (int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
         thread_idx < shard_size * float_feature_len;
         thread_idx += blockDim.x * gridDim.x) {
        int64_t local_id = thread_idx / float_feature_len;
        int32_t feat_offset = thread_idx % float_feature_len;
        int32_t node_id = shard_ids[local_id];
        if (node_id >= 0) {
            coordinate_shard[thread_idx] =
                cpu_float_features[int64_t(node_id) * float_feature_len + feat_offset];
        }
    }
}

__global__ void coord_shard_lookup(
    float** coordinate_shards,
    int64_t* shard_offsets,
    int32_t shard_count,
    int32_t* root_positions,
    int32_t float_feature_len,
    int32_t* sampled_ids,
    int32_t* node_counter,
    float* dst_float_buffer,
    int32_t op_id)
{
    // 当前的OP处理的节点在 sampled_ids 中的起始偏移
    int32_t node_off = node_counter[(op_id % INTRABATCH_CON) * 2];
    // 当前这一步要处理多少个节点
    int32_t batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
    for (int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
         thread_idx < int64_t(batch_size) * float_feature_len;
         thread_idx += blockDim.x * gridDim.x) {
        // 当前 batch 内第几个节点
        int32_t local_node = thread_idx / float_feature_len;
        // 这个节点的第几维特征
        int32_t feat_offset = thread_idx % float_feature_len;
        // 拿到的是全局节点编号，不是 shard 内部编号
        int32_t global_id = sampled_ids[node_off + local_node];
        // 计算输出写入位置
        int64_t dst_offset = int64_t(node_off + local_node) * float_feature_len + feat_offset;
        if (global_id < 0) {
            dst_float_buffer[dst_offset] = 0.0f;
            continue;
        }
        // global_id 映射成 shard_idx
        int32_t gidx = root_positions[global_id];
        // gidx小于0代表没有写入unified shard
        if (gidx < 0) {
            dst_float_buffer[dst_offset] = 0.0f;
            continue;
        }
        int32_t shard = 0;
        // 因为每个shard存储特征的数量不是相同的，所以需要累加比对，确定索引在哪个shard区间
        while (shard + 1 < shard_count && gidx >= shard_offsets[shard + 1]) {
            shard++;
        }
        // 算出 shard 内部的 offset
        int64_t local_id = int64_t(gidx) - shard_offsets[shard];
        // 将shard上的读取真实特征
        dst_float_buffer[dst_offset] =
            coordinate_shards[shard][local_id * float_feature_len + feat_offset];
    }
}



void mmap_cache_read(std::string &cache_file, std::vector<int32_t>& cache_map){
    int64_t t_idx = 0;
    int32_t fd = open(cache_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<cache_file<<std::endl;
    }
    // int64_t buf_len = lseek(fd, 0, SEEK_END);
    int64_t buf_len = int64_t(int64_t(cache_map.size()) * 4);
    const int32_t* buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        cache_map[t_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}


#endif
