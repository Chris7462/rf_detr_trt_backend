#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/core.hpp>

#include "rf_detr_trt_backend/rf_detr_trt_backend.hpp"
#include "internal/trt_logger.hpp"
#include "internal/cuda_raii.hpp"

namespace rf_detr_trt_backend
{

class RFDetrTrtBackend::Impl
{
public:
  ~Impl();
  void initialize(const std::string & engine_path, const RFDetrTrtBackend::Config & config);
  Detections infer(const cv::Mat & image, const RFDetrTrtBackend::Config & config);

  // Model geometry, resolved from the engine's own tensor shapes in
  // find_tensor_names(). Only meaningful after initialize() has succeeded.
  int input_height() const noexcept {return input_height_;}
  int input_width() const noexcept {return input_width_;}
  int num_queries() const noexcept {return num_queries_;}
  // Dense foreground class count (num_classes_with_bg_ - 1, excludes the
  // background/no-object slot) - matches what RFDetrTrtBackend::Config used
  // to call "num_classes" before it was removed as a configurable field.
  int num_classes() const noexcept {return num_classes_with_bg_ - 1;}

private:
  void initialize_engine(const std::string & engine_path, const RFDetrTrtBackend::Config & config);
  void find_tensor_names();
  void initialize_memory();
  void initialize_streams();
  void warmup_engine(const RFDetrTrtBackend::Config & config);
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

  // Model geometry, resolved from the engine's own tensor shapes in
  // find_tensor_names() - see the class-level comment on the accessors
  // above. Never sourced from Config.
  int input_height_ = 0;
  int input_width_ = 0;
  int num_queries_ = 0;
  int num_classes_with_bg_ = 0;  // includes the background/no-object slot

  struct MemoryBuffers
  {
    internal::DevPtr device_input_raw;
    internal::DevPtr device_preprocessed;
    internal::DevPtr device_dets;
    internal::DevPtr device_labels;
    internal::HostPtr pinned_dets;
    internal::HostPtr pinned_labels;
  } buffers_;

  internal::StreamPtr stream_;
};

}
