#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/core.hpp>

#include "rf_detr_trt_backend/rf_detr_trt_backend.hpp"
#include "internal/trt_logger.hpp"

namespace rf_detr_trt_backend
{

class RFDetrTrtBackend::Impl
{
public:
  ~Impl();
  void initialize(const std::string & engine_path, const RFDetrTrtBackend::Config & config);
  Detections infer(const cv::Mat & image, const RFDetrTrtBackend::Config & config);

private:
  void initialize_engine(const std::string & engine_path, const RFDetrTrtBackend::Config & config);
  void find_tensor_names();
  void initialize_memory(const RFDetrTrtBackend::Config & config);
  void initialize_streams();
  void warmup_engine(const RFDetrTrtBackend::Config & config);
  void cleanup() noexcept;
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  void upload_and_preprocess(
    const cv::Mat & image, const RFDetrTrtBackend::Config & config, cudaStream_t stream) const;

private:
  std::unique_ptr<internal::Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::string input_name_;
  std::string dets_name_;
  std::string labels_name_;
  size_t input_size_ = 0;
  size_t preprocessed_size_ = 0;
  size_t dets_size_ = 0;
  size_t labels_size_ = 0;

  struct MemoryBuffers
  {
    unsigned char * device_input_raw;
    float * device_preprocessed;
    float * device_dets;
    float * device_labels;
    float * pinned_dets;
    float * pinned_labels;

    MemoryBuffers()
    : device_input_raw(nullptr), device_preprocessed(nullptr),
      device_dets(nullptr), device_labels(nullptr),
      pinned_dets(nullptr), pinned_labels(nullptr) {}
  } buffers_;

  cudaStream_t stream_ = nullptr;
};

}
