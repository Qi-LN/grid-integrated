#include "feature_storage.cuh"
#include "feature_storage_impl.cuh"
#include <iostream>

#include <unordered_set>
#include <algorithm>
#include <random>
#include <assert.h>
#include <unistd.h>

class CompleteFeatureStorage : public FeatureStorage{
public: 
    CompleteFeatureStorage(){
    }

    virtual ~CompleteFeatureStorage(){};

    void Build(BuildInfo* info, int in_memory_mode) override {
        int32_t partition_count = info->partition_count;
        total_num_nodes_ = info->total_num_nodes;
        float_feature_len_ = info->float_feature_len;
        float* host_float_feature = info->host_float_feature;

        if(in_memory_mode){
            cudaHostGetDevicePointer(&float_feature_, host_float_feature, 0);
        }
        cudaCheckError();

        training_set_num_.resize(partition_count);
        training_set_ids_.resize(partition_count);
        training_labels_.resize(partition_count);

        validation_set_num_.resize(partition_count);
        validation_set_ids_.resize(partition_count);
        validation_labels_.resize(partition_count);

        testing_set_num_.resize(partition_count);
        testing_set_ids_.resize(partition_count);
        testing_labels_.resize(partition_count);
        inference_set_num_.resize(partition_count);
        inference_shard_num_.resize(partition_count);
        inference_set_ids_.resize(partition_count);

        partition_count_ = partition_count;

        for(int32_t i = 0; i < partition_count_; i++){
            int32_t part_id = i;
            training_set_num_[part_id] = info->training_set_num[part_id];
            validation_set_num_[part_id] = info->validation_set_num[part_id];
            testing_set_num_[part_id] = info->testing_set_num[part_id];
            inference_set_num_[part_id] = info->inference_set_num[part_id];
            inference_shard_num_[part_id] = info->inference_shard_num.empty()
                ? info->inference_set_num[part_id]
                : info->inference_shard_num[part_id];

            cudaSetDevice(part_id);
            cudaCheckError();

            int32_t* train_ids;
            cudaMalloc(&train_ids, training_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(train_ids, info->training_set_ids[part_id].data(), training_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            training_set_ids_[part_id] = train_ids;
            cudaCheckError();

            int32_t* valid_ids;
            cudaMalloc(&valid_ids, validation_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(valid_ids, info->validation_set_ids[part_id].data(), validation_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            validation_set_ids_[part_id] = valid_ids;
            cudaCheckError();

            int32_t* test_ids;
            cudaMalloc(&test_ids, testing_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(test_ids, info->testing_set_ids[part_id].data(), testing_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            testing_set_ids_[part_id] = test_ids;
            cudaCheckError();

            int32_t* train_labels;
            cudaMalloc(&train_labels, training_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(train_labels, info->training_labels[part_id].data(), training_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            training_labels_[part_id] = train_labels;
            cudaCheckError();

            int32_t* valid_labels;
            cudaMalloc(&valid_labels, validation_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(valid_labels, info->validation_labels[part_id].data(), validation_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            validation_labels_[part_id] = valid_labels;
            cudaCheckError();

            int32_t* test_labels;
            cudaMalloc(&test_labels, testing_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(test_labels, info->testing_labels[part_id].data(), testing_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            testing_labels_[part_id] = test_labels;
            cudaCheckError();

            // 保存当前 part_id 对应的 infer 节点 ID 列表
            int32_t* infer_ids = nullptr;
            if (inference_shard_num_[part_id] > 0) {
                cudaMalloc(&infer_ids, inference_shard_num_[part_id] * sizeof(int32_t));
                cudaMemcpy(infer_ids, info->inference_set_ids[part_id].data(), inference_shard_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
                cudaCheckError();
            }
            inference_set_ids_[part_id] = infer_ids;

        }

    };

    void Finalize() override {
        cudaFreeHost(float_feature_);
        for(int32_t i = 0; i < partition_count_; i++){
            cudaSetDevice(i);
            cudaFree(training_set_ids_[i]);
            cudaFree(validation_set_ids_[i]);
            cudaFree(testing_set_ids_[i]);
            cudaFree(training_labels_[i]);
            cudaFree(validation_labels_[i]);
            cudaFree(testing_labels_[i]);
            if (inference_set_ids_[i] != nullptr) {
                cudaFree(inference_set_ids_[i]);
            }
        }
    }

    int32_t* GetTrainingSetIds(int32_t part_id) const override {
        return training_set_ids_[part_id];
    }
    int32_t* GetValidationSetIds(int32_t part_id) const override {
        return validation_set_ids_[part_id];
    }
    int32_t* GetTestingSetIds(int32_t part_id) const override {
        return testing_set_ids_[part_id];
    }
    int32_t* GetInferenceSetIds(int32_t part_id) const override {
        return inference_set_ids_[part_id];
    }

    int32_t* GetTrainingLabels(int32_t part_id) const override {
        return training_labels_[part_id];
    };
    int32_t* GetValidationLabels(int32_t part_id) const override {
        return validation_labels_[part_id];
    }
    int32_t* GetTestingLabels(int32_t part_id) const override {
        return testing_labels_[part_id];
    }

    int32_t TrainingSetSize(int32_t part_id) const override {
        return training_set_num_[part_id];
    }
    int32_t ValidationSetSize(int32_t part_id) const override {
        return validation_set_num_[part_id];
    }
    int32_t TestingSetSize(int32_t part_id) const override {
        return testing_set_num_[part_id];
    }
    int32_t InferenceSetSize(int32_t part_id) const override {
        return inference_set_num_[part_id];
    }
    int32_t InferenceShardSize(int32_t part_id) const override {
        return inference_shard_num_[part_id];
    }

    int32_t TotalNodeNum() const override {
        return total_num_nodes_;
    }

    float* GetAllFloatFeature() const override {
        return float_feature_;
    }
    int32_t GetFloatFeatureLen() const override {
        return float_feature_len_;
    }

    void IOSubmit(int32_t* sampled_ids, int32_t* cache_index,
                  int32_t* node_counter, float* dst_float_buffer,
                  int32_t op_id, int32_t dev_id, cudaStream_t strm_hdl) override {
        //TODO
    }

    void IOComplete() override {
        //TODO
    }

private:
    std::vector<int> training_set_num_;
    std::vector<int> validation_set_num_;
    std::vector<int> testing_set_num_;

    std::vector<int32_t*> training_set_ids_;
    std::vector<int32_t*> validation_set_ids_;
    std::vector<int32_t*> testing_set_ids_;
    std::vector<int32_t*> inference_set_ids_;

    std::vector<int32_t*> training_labels_;
    std::vector<int32_t*> validation_labels_;
    std::vector<int32_t*> testing_labels_;
    std::vector<int> inference_set_num_;
    std::vector<int> inference_shard_num_;

    int32_t partition_count_;
    int32_t total_num_nodes_;
    float* float_feature_;
    int32_t float_feature_len_;

    friend FeatureStorage* NewCompleteFeatureStorage();
};

extern "C" 
FeatureStorage* NewCompleteFeatureStorage(){
    CompleteFeatureStorage* ret = new CompleteFeatureStorage();
    return ret;
}
