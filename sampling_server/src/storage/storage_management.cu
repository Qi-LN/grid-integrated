#include "storage_management.cuh"
#include "storage_management_impl.cuh"
#include "system_config.cuh"

#include <fstream>
#include <sstream>
#include <unordered_set>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>

namespace {

int32_t resolve_owner_by_position(int64_t position, int32_t shard_count, int64_t total_count) {
    if (position < 0 || position >= total_count || shard_count <= 0 || total_count <= 0) {
        return -1;
    }
    int64_t owner = (position * shard_count) / total_count;
    if (owner >= shard_count) {
        owner = shard_count - 1;
    }
    return static_cast<int32_t>(owner);
}

// 检查某个路径对应的文件或目录是否存在
bool file_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}

// 把一个二进制文件按 int32_t 数组读出来，返回一个 std::vector<int32_t>
// 布局：[int32_t][int32_t][int32_t]...
std::vector<int32_t> read_int32_binary_file(const std::string& path) {
    std::vector<int32_t> result;
    if (!file_exists(path)) {
        return result;
    }
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        return result;
    }
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size <= 0) {
        return result;
    }
    result.resize(static_cast<size_t>(size) / sizeof(int32_t));
    in.read(reinterpret_cast<char*>(result.data()), size);
    return result;
}


// 尝试从一个文本文件里读取一个 int32_t 整数；如果文件不存在或打不开，就返回默认值
// 干啥的？？？
int32_t read_optional_int32_text_file(const std::string& path, int32_t default_value) {
    if (!file_exists(path)) {
        return default_value;
    }
    std::ifstream in(path);
    if (!in.is_open()) {
        return default_value;
    }
    int32_t value = default_value;
    in >> value;
    return value;
}

}

void StorageManagement::EnableP2PAccess(){
    int32_t device_count = -1;
    cudaGetDeviceCount(&device_count);
    for(int32_t i = 0; i < device_count; i++){
        cudaSetDevice(i);
        cudaCheckError();
        for(int32_t j = 0; j < device_count; j++){
          if(j != i){
            int32_t accessible = 0;
            cudaDeviceCanAccessPeer(&accessible, i, j);
            cudaCheckError();
            if(accessible){
              cudaDeviceEnablePeerAccess(j, 0);
              cudaCheckError();
            }
          }
        }
    }
}

void StorageManagement::ConfigPartition(BuildInfo* info, int32_t partition_count){
    info->partition_count = partition_count;
}

void StorageManagement::ReadMetaFIle(BuildInfo* info){
    std::istringstream iss;
    std::string buff;
    std::ifstream Metafile("./meta_config");
    if(!Metafile.is_open()){
     std::cout<<"unable to open meta config file"<<"\n";
    }
    getline(Metafile, buff);
    iss.clear();
    iss.str(buff);

    // 默认初始化
    partition_meta_path_ = "-";
    node_rank_ = 0;
    num_nodes_ = 1;
    local_gpu_number_ = info->partition_count;
    infer_step_override_ = -1;

    if(in_memory_mode_){
        iss >> dataset_path_;
        std::cout<<"Dataset path:       "<<dataset_path_<<"\n";
        iss >> raw_batch_size_;
        std::cout<<"Raw Batchsize:      "<<raw_batch_size_<<"\n";
        info->raw_batch_size = raw_batch_size_;
        iss >> node_num_;
        std::cout<<"Graph nodes num:    "<<node_num_<<"\n";
        iss >> edge_num_;
        std::cout<<"Graph edges num:    "<<edge_num_<<"\n";
        iss >> float_feature_len_;
        std::cout<<"Feature dim:        "<<float_feature_len_<<"\n";
        iss >> training_set_num_;
        std::cout<<"Training set num:   "<<training_set_num_<<"\n";
        iss >> validation_set_num_;
        std::cout<<"Validation set num: "<<validation_set_num_<<"\n";
        iss >> testing_set_num_;
        std::cout<<"Testing set num:    "<<testing_set_num_<<"\n";
        iss >> cache_memory_;
        std::cout<<"Cache memory:       "<<cache_memory_<<"\n";
        iss >> epoch_;
        std::cout<<"Train epoch:        "<<epoch_<<"\n";
        info->epoch = epoch_;
        if (iss >> serve_mode_) {
            std::cout<<"Serve mode:         "<<serve_mode_<<"\n";
        } else {
            serve_mode_ = SERVE_TRAIN;
            iss.clear();
        }
        if (iss >> rootset_path_) {
            std::cout<<"Rootset path:       "<<rootset_path_<<"\n";
        } else {
            rootset_path_ = "-";
            iss.clear();
        }
        if (iss >> partition_meta_path_) {
            std::cout<<"Partition meta:     "<<partition_meta_path_<<"\n";
        } else {
            partition_meta_path_ = "-";
            iss.clear();
        }
        if (iss >> node_rank_) {
            std::cout<<"Node rank:          "<<node_rank_<<"\n";
        } else {
            node_rank_ = 0;
            iss.clear();
        }
        if (iss >> num_nodes_) {
            std::cout<<"Num nodes:          "<<num_nodes_<<"\n";
        } else {
            num_nodes_ = 1;
            iss.clear();
        }
        if (iss >> local_gpu_number_) {
            std::cout<<"Local GPU num:      "<<local_gpu_number_<<"\n";
        } else {
            local_gpu_number_ = info->partition_count;
            iss.clear();
        }
        if (iss >> infer_step_override_) {
            std::cout<<"Infer step override:"<<infer_step_override_<<"\n";
        } else {
            infer_step_override_ = -1;
            iss.clear();
        }
    }else{
        iss >> dataset_path_;
        std::cout<<"Dataset path:       "<<dataset_path_<<"\n";
        iss >> raw_batch_size_;
        std::cout<<"Raw Batchsize:      "<<raw_batch_size_<<"\n";
        info->raw_batch_size = raw_batch_size_;
        iss >> node_num_;
        std::cout<<"Graph nodes num:    "<<node_num_<<"\n";
        iss >> edge_num_;
        std::cout<<"Graph edges num:    "<<edge_num_<<"\n";
        iss >> float_feature_len_;
        std::cout<<"Feature dim:        "<<float_feature_len_<<"\n";
        iss >> training_set_num_;
        std::cout<<"Training set num:   "<<training_set_num_<<"\n";
        iss >> validation_set_num_;
        std::cout<<"Validation set num: "<<validation_set_num_<<"\n";
        iss >> testing_set_num_;
        std::cout<<"Testing set num:    "<<testing_set_num_<<"\n";
        iss >> cache_memory_;
        std::cout<<"Cache memory:       "<<cache_memory_<<"\n";
        iss >> epoch_;
        std::cout<<"Train epoch:        "<<epoch_<<"\n";
        info->epoch = epoch_;
        if (iss >> serve_mode_) {
            std::cout<<"Serve mode:         "<<serve_mode_<<"\n";
        } else {
            serve_mode_ = SERVE_TRAIN;
            iss.clear();
        }
        if (iss >> rootset_path_) {
            std::cout<<"Rootset path:       "<<rootset_path_<<"\n";
        } else {
            rootset_path_ = "-";
            iss.clear();
        }
        if (iss >> partition_meta_path_) {
            std::cout<<"Partition meta:     "<<partition_meta_path_<<"\n";
        } else {
            partition_meta_path_ = "-";
            iss.clear();
        }
        if (iss >> node_rank_) {
            std::cout<<"Node rank:          "<<node_rank_<<"\n";
        } else {
            node_rank_ = 0;
            iss.clear();
        }
        if (iss >> num_nodes_) {
            std::cout<<"Num nodes:          "<<num_nodes_<<"\n";
        } else {
            num_nodes_ = 1;
            iss.clear();
        }
        if (iss >> local_gpu_number_) {
            std::cout<<"Local GPU num:      "<<local_gpu_number_<<"\n";
        } else {
            local_gpu_number_ = info->partition_count;
            iss.clear();
        }
        if (iss >> infer_step_override_) {
            std::cout<<"Infer step override:"<<infer_step_override_<<"\n";
        } else {
            infer_step_override_ = -1;
            iss.clear();
        }
        iss >> partition_;
        std::cout<<"Partition?:         "<<partition_<<"\n";
        iss >> num_ssd_;
        std::cout<<"SSD Num?:           "<<num_ssd_<<"\n";
        iss >> num_queues_per_ssd_;
        std::cout<<"Q/SSD    ?:         "<<num_queues_per_ssd_<<"\n";
        iss >> cpu_cache_capacity_;
        std::cout<<"CPU Cache Capacity: "<<cpu_cache_capacity_<<"\n";
        iss >> gpu_cache_capacity_;
        std::cout<<"GPU Cache Capacity: "<<gpu_cache_capacity_<<"\n";
    }
    info->serve_mode = serve_mode_;
    info->rootset_path = rootset_path_;
    info->partition_meta_path = partition_meta_path_;
    info->node_rank = node_rank_;
    info->num_nodes = num_nodes_;
    info->local_gpu_number = local_gpu_number_;
    info->infer_step_override = infer_step_override_;
}

void StorageManagement::LoadGraph(BuildInfo* info){

    int32_t node_num = node_num_;
    int64_t edge_num = edge_num_;
    info->total_edge_num = edge_num;
    info->cache_edge_num = cache_edge_num_;

    //uva
    cudaHostAlloc(&(info->csr_node_index), int64_t(int64_t(node_num + 1)*sizeof(int64_t)), cudaHostAllocMapped);
    cudaHostAlloc(&(info->csr_dst_node_ids), int64_t(int64_t(edge_num) * sizeof(int32_t)), cudaHostAllocMapped);
    std::string edge_src_path = dataset_path_ + "edge_src";
    std::string edge_dst_path = dataset_path_ + "edge_dst";

    mmap_indptr_read(edge_src_path, info->csr_node_index);
    mmap_indices_read(edge_dst_path, info->csr_dst_node_ids);

    // compute max degree for buffer allocation
    max_degree_ = 0;
    for (int32_t node_id = 0; node_id < node_num; ++node_id) {
        int64_t degree = info->csr_node_index[node_id + 1] - info->csr_node_index[node_id];
        if (degree > max_degree_) {
            max_degree_ = degree;
        }
    }
    info->max_degree = max_degree_;
    std::cout<<"Max degree:         "<<max_degree_<<"\n";
}


void StorageManagement::LoadFeature(BuildInfo* info){

    int32_t partition_count = info->partition_count;

    int32_t node_num = node_num_;
    int32_t nf = float_feature_len_;

    info->numElems = uint64_t(node_num) * nf;

    (info->training_set_ids).resize(partition_count);
    (info->training_labels).resize(partition_count);
    (info->validation_set_ids).resize(partition_count);
    (info->validation_labels).resize(partition_count);
    (info->testing_set_ids).resize(partition_count);
    (info->testing_labels).resize(partition_count);
    (info->inference_set_ids).resize(partition_count);

    std::string training_path = dataset_path_  + "trainingset";
    std::string validation_path = dataset_path_  + "validationset";
    std::string testing_path = dataset_path_  + "testingset";
    // std::string training_path = dataset_path_  + "train_ids";
    // std::string validation_path = dataset_path_  + "valid_ids";
    // std::string testing_path = dataset_path_  + "test_ids";
    std::string features_path = dataset_path_ + "features";
    std::string labels_path = dataset_path_ + "labels";
    // std::string labels_path = dataset_path_ + "labels_raw";

    std::string partition_path = dataset_path_ + "partition";

    std::vector<int32_t> training_ids;
    training_ids.resize(training_set_num_);
    std::vector<int32_t> validation_ids;
    validation_ids.resize(validation_set_num_);
    std::vector<int32_t> testing_ids;
    testing_ids.resize(testing_set_num_);
    std::vector<int32_t> all_labels;
    all_labels.resize(node_num);
    int32_t* partition_index = (int32_t*)malloc(int64_t(node_num) * sizeof(int32_t));
    float* host_float_feature = nullptr;

    mmap_trainingset_read(training_path, training_ids);
    mmap_trainingset_read(validation_path, validation_ids);
    mmap_trainingset_read(testing_path, testing_ids);
    if(in_memory_mode_){
        cudaHostAlloc(&host_float_feature, int64_t(int64_t(int64_t(node_num) * nf) * sizeof(float)), cudaHostAllocMapped);
        mmap_features_read(features_path, host_float_feature);
    }
    mmap_labels_read(labels_path, all_labels);

    int32_t fdret = mmap_partition_read(partition_path, partition_index);

    std::cout<<"Finish Reading All Files\n";
    // partition nodes，把顶点分配到每个gpu内的集合中。
    // training_ids->info->training_set_ids

    int trainingset_count = 0;
    for(int32_t i = 0; i < training_set_num_; i+=1){
        int32_t tid = training_ids[i];
        int32_t part_id;
        if(fdret >= 0){
            part_id = partition_index[tid];
        }else{
            part_id = tid % partition_count;
        }
        if(part_id < partition_count){
            (info->training_set_ids[part_id]).push_back(tid);
            trainingset_count ++ ;
        }
    }
    std::cout<<"training set count "<<trainingset_count<<"\n";

    for(int32_t i = 0; i < validation_set_num_; i++){
        int32_t tid = validation_ids[i];
        int32_t part_id = tid % partition_count;

        if(part_id < partition_count){
            (info->validation_set_ids[part_id]).push_back(tid);
        }
    }

    for(int32_t i = 0; i < testing_set_num_; i++){
        int32_t tid = testing_ids[i];
        int32_t part_id = tid % partition_count;

        if(part_id < partition_count){
            (info->testing_set_ids[part_id]).push_back(tid);
        }
    }
    free(partition_index);

    //partition labels
    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < (int32_t)info->training_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->training_set_ids[part_id][i]];
            info->training_labels[part_id].push_back(ts_label);
        }
        info->training_set_num.push_back(info->training_set_ids[part_id].size());
    }
    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < (int32_t)info->validation_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->validation_set_ids[part_id][i]];
            info->validation_labels[part_id].push_back(ts_label);
        }
        info->validation_set_num.push_back(info->validation_set_ids[part_id].size());
    }
    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < (int32_t)info->testing_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->testing_set_ids[part_id][i]];
            info->testing_labels[part_id].push_back(ts_label);
        }
        info->testing_set_num.push_back(info->testing_set_ids[part_id].size());
    }

    info->host_float_feature = host_float_feature;
    info->float_feature_len = float_feature_len_;
    info->total_num_nodes = node_num_;

    // 在推理服务模式下，从磁盘上的 partition metadata 中读取“每个 GPU 负责哪些 owner roots、哪些 halo nodes”，并做严格一致性检查
    bool loaded_partition_meta = false;
    if (serve_mode_ == SERVE_INFER && partition_meta_path_ != "-" && !partition_meta_path_.empty()) {
        std::string infer_step_path = partition_meta_path_ + "/infer_steps.txt";
        int32_t infer_steps_from_meta = read_optional_int32_text_file(infer_step_path, infer_step_override_);
        if (infer_steps_from_meta > 0) {
            info->infer_step_override = infer_steps_from_meta;
            infer_step_override_ = infer_steps_from_meta;
        } else if (num_nodes_ > 1) {
            std::cout << "missing infer_steps.txt for multi-node partition metadata\n";
            exit(EXIT_FAILURE);
        }

        // 构造当前节点 metadata 目录
        std::string node_meta_dir = partition_meta_path_ + "/node_" + std::to_string(node_rank_);
        std::unordered_set<int32_t> seen_local_nodes;
        int64_t total_owner_roots = 0;
        int64_t total_halo_nodes = 0;
        for (int32_t part_id = 0; part_id < partition_count; ++part_id) {
            std::string owner_path = node_meta_dir + "/gpu_" + std::to_string(part_id) + "_owner_roots.bin";
            std::string halo_path = node_meta_dir + "/gpu_" + std::to_string(part_id) + "_halo_nodes.bin";
            std::vector<int32_t> owner_roots = read_int32_binary_file(owner_path);
            std::vector<int32_t> halo_nodes = read_int32_binary_file(halo_path);

            // 处理 owner_roots
            std::unordered_set<int32_t> owner_set;
            for (int32_t node_id : owner_roots) {
                if (node_id < 0 || node_id >= node_num_) {
                    std::cout << "invalid owner root id in metadata: " << node_id << "\n";
                    exit(EXIT_FAILURE);
                }
                // 放进 owner_set
                owner_set.insert(node_id);
                // 检查当前 node 内是否跨 GPU 重复,避免这个 node_id 之前已经分给本节点的其他 GPU 了。
                if (!seen_local_nodes.insert(node_id).second) {
                    std::cout << "duplicate local shard node across GPUs: " << node_id << "\n";
                    exit(EXIT_FAILURE);
                }
                // 把 owner root 记录到该 GPU 的 inference set 里。
                // 没区分 owner 和 halo 的存储位置，都会被放进同一个 inference_set_ids[part_id] 中，owner 先放，halo 后放。
                info->inference_set_ids[part_id].push_back(node_id);
            }
            // 处理 halo_nodes
            for (int32_t node_id : halo_nodes) {
                if (node_id < 0 || node_id >= node_num_) {
                    std::cout << "invalid halo node id in metadata: " << node_id << "\n";
                    exit(EXIT_FAILURE);
                }
                // 检查 halo 和 owner 是否重叠
                if (owner_set.find(node_id) != owner_set.end()) {
                    std::cout << "owner/halo overlap in metadata for gpu " << part_id << ": " << node_id << "\n";
                    exit(EXIT_FAILURE);
                }
                // 检查是否跨 GPU 重复
                if (!seen_local_nodes.insert(node_id).second) {
                    std::cout << "duplicate local shard node across GPUs: " << node_id << "\n";
                    exit(EXIT_FAILURE);
                }
                // 追加到 inference_set_ids
                info->inference_set_ids[part_id].push_back(node_id);
            }
            // 记录每个 GPU 的 owner 数
            info->inference_set_num.push_back(static_cast<int32_t>(owner_roots.size()));
            // 记录每个 GPU 的总 shard 数
            info->inference_shard_num.push_back(static_cast<int32_t>(owner_roots.size() + halo_nodes.size()));
            total_owner_roots += owner_roots.size();
            total_halo_nodes += halo_nodes.size();
        }
        std::cout << "Inference owner roots: " << total_owner_roots << "\n";
        std::cout << "Inference halo nodes:  " << total_halo_nodes << "\n";
        loaded_partition_meta = true;
    }
    // 如果前面没有成功加载 partition metadata，那么就退而求其次，从一个 rootset 文件里读出一串 root 节点，并按“位置”把这些 root 分配给各个 GPU。
    // partition inference root. 文件->info->inference_set_ids. TODO:修改为按GPU加载
    if (!loaded_partition_meta && serve_mode_ == SERVE_INFER && rootset_path_ != "-" && !rootset_path_.empty()) {
        int32_t root_fd = open(rootset_path_.c_str(), O_RDONLY);
        if (root_fd == -1) {
            std::cout<<"cannout open file: "<<rootset_path_<<"\n";
        } else {
            int64_t root_buf_len = lseek(root_fd, 0, SEEK_END);
            const int32_t* root_buf = (int32_t*)mmap(NULL, root_buf_len, PROT_READ, MAP_PRIVATE, root_fd, 0);
            const int32_t* root_end = root_buf + root_buf_len / sizeof(int32_t);
            int64_t root_total = root_buf_len / sizeof(int32_t);
            int64_t root_count = 0;
            int64_t root_position = 0;
            while (root_buf < root_end) {
                int32_t root_id = *root_buf++;
                int32_t owner = resolve_owner_by_position(root_position, partition_count, root_total);
                if (root_id >= 0 && root_id < node_num_ && owner >= 0 && owner < partition_count) {
                    info->inference_set_ids[owner].push_back(root_id);
                    root_count++;
                }
                root_position++;
            }
            close(root_fd);
            std::cout<<"Inference roots:    "<<root_count<<"\n";
            if (root_total != node_num_) {
                std::cout<<"Warning: root file count "<<root_total<<" != node count "<<node_num_<<"\n";
            }
        }
    }
    // 兜底初始化，如果前面没有给 inference_shard_num 赋值，那么就默认把每个 partition 的 inference_set_ids 全都视为“owner 集合”，同时也把它当成完整 shard。
    if (info->inference_shard_num.empty()) {
        for (int32_t part_id = 0; part_id < partition_count; ++part_id) {
            info->inference_set_num.push_back(info->inference_set_ids[part_id].size());
            info->inference_shard_num.push_back(info->inference_set_ids[part_id].size());
        }
    }
}

void StorageManagement::Initialze(int32_t partition_count, int32_t in_memory_mode){

    in_memory_mode_ = in_memory_mode;

    BuildInfo* info = new BuildInfo();

    EnableP2PAccess();

    ConfigPartition(info, partition_count);

    ReadMetaFIle(info);

    LoadGraph(info);

    LoadFeature(info);

    env_ = NewIPCEnv(partition_count);
    env_ -> Coordinate(info);

    feature_ = NewCompleteFeatureStorage();
    feature_ -> Build(info, in_memory_mode_);

    graph_ = NewCompleteGraphStorage();
    graph_ -> Build(info);

    cudaCheckError();

    cache_ = new UnifiedCache();

    int32_t train_step = env_->GetTrainStep();

    cudaSetDevice(0);
    cache_ -> Initialize(cache_memory_, float_feature_len_, train_step, partition_count, cpu_cache_capacity_, gpu_cache_capacity_);
    if (serve_mode_ == SERVE_INFER) {
        // 1. 构建unified shard + 映射表  2. 把每个GPU上对应的coordinate shard的起始global id和size发布到IPCEnv里
        cache_->InitializeCoordinateStore(feature_);
        env_->PublishCoordinateShards(cache_);
    }
    cudaSetDevice(0);
    std::cout<<"Storage Initialized\n";
}

GraphStorage* StorageManagement::GetGraph(){
    return graph_;
}

FeatureStorage* StorageManagement::GetFeature(){
    return feature_;
}

UnifiedCache* StorageManagement::GetCache(){
    return cache_;
}

IPCEnv* StorageManagement::GetIPCEnv(){
    return env_;
}
