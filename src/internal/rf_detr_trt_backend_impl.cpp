// C++ standard library includes
#include <fstream>
#include <iostream>
#include <stdexcept>

// local headers
#include "internal/rf_detr_trt_backend_impl.hpp"
#include "internal/cuda_check.hpp"
#include "internal/trt_logger.hpp"
#include "internal/preprocess_kernel.cuh"
#include "internal/postprocess.hpp"


namespace rf_detr_trt_backend
{

namespace
{

// Allocate device memory and immediately wrap it in an owning DevPtr, so
// there is no window where a raw, unmanaged pointer exists in caller scope.
internal::DevPtr cuda_malloc_dev(size_t bytes)
{
  void * p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  return internal::DevPtr(p);
}

// Same idea for pinned host memory.
internal::HostPtr cuda_malloc_host(size_t bytes)
{
  void * p = nullptr;
  CUDA_CHECK(cudaMallocHost(&p, bytes));
  return internal::HostPtr(p);
}

} // namespace

RFDetrTrtBackend::Impl::~Impl() = default;

void RFDetrTrtBackend::Impl::initialize(
  const std::string & engine_path, const RFDetrTrtBackend::Config & config)
{
  initialize_engine(engine_path, config);
  find_tensor_names();
  initialize_memory(config);
  initialize_streams();
  // NOTE: no initialize_constants() step here, unlike fcn_trt_backend.
  // mean/std are passed as per-call kernel arguments (see
  // preprocess_kernel.cu), not copied into __constant__ memory, so there is
  // no one-time constant-memory setup to do.
  warmup_engine(config);
}

Detections RFDetrTrtBackend::Impl::infer(
  const cv::Mat & image, const RFDetrTrtBackend::Config & config)
{
  // Hard-require the caller (the ROS2 node) to have already produced an
  // exactly-sized, contiguous BGR8 image. No resizing happens here - see
  // RFDetrTrtBackend::infer()'s doc comment for the rationale.
  if (image.rows != config.height || image.cols != config.width ||
    image.type() != CV_8UC3)
  {
    throw std::invalid_argument(
      "RFDetrTrtBackend::infer(): image must be exactly " +
      std::to_string(config.height) + "x" + std::to_string(config.width) +
      " CV_8UC3, got " + std::to_string(image.rows) + "x" +
      std::to_string(image.cols) + " (type " + std::to_string(image.type()) + ")");
  }
  if (!image.isContinuous()) {
    throw std::invalid_argument(
      "RFDetrTrtBackend::infer(): image must be continuous "
      "(e.g. not a non-contiguous ROI view) - clone() it first");
  }

  // Upload + normalize directly into GPU memory
  upload_and_preprocess(image, config, stream_.get());

  // Run inference
  if (!context_->enqueueV3(stream_.get())) {
    throw internal::TensorRTException("Failed to enqueue inference");
  }

  // Copy both outputs back to pinned host memory for CPU-side decode
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_dets.get(), buffers_.device_dets.get(),
    dets_size_, cudaMemcpyDeviceToHost, stream_.get()));
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_labels.get(), buffers_.device_labels.get(),
    labels_size_, cudaMemcpyDeviceToHost, stream_.get()));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

  // Decode on CPU. Boxes come out in the square input's own pixel space
  // (config.width x config.height) - un-letterboxing to the original image
  // is the caller's responsibility, one layer up.
  internal::PostprocessParams params;
  params.num_queries = config.num_queries;
  params.num_classes_with_bg = config.num_classes + 1;
  params.topk = config.num_queries;
  params.threshold = config.score_threshold;
  params.bg_class_index = internal::kBackgroundClassIndex;

  return internal::decode_detections(
    static_cast<const float *>(buffers_.pinned_dets.get()),
    static_cast<const float *>(buffers_.pinned_labels.get()),
    config.width, config.height, params);
}

void RFDetrTrtBackend::Impl::initialize_engine(
  const std::string & engine_path, const RFDetrTrtBackend::Config & config)
{
  // Initialize logger
  logger_ = std::make_unique<internal::Logger>(internal::to_trt_severity(config.log_level));

  auto engine_data = load_engine_file(engine_path);

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw internal::TensorRTException("Failed to create TensorRT runtime");
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    throw internal::TensorRTException("Failed to deserialize CUDA engine");
  }

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
    engine_->createExecutionContext());
  if (!context_) {
    throw internal::TensorRTException("Failed to create execution context");
  }
}

void RFDetrTrtBackend::Impl::find_tensor_names()
{
  bool found_input = false, found_dets = false, found_labels = false;
  std::vector<std::string> unrecognized_outputs;

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = tensor_name;
      found_input = true;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      const std::string name(tensor_name);
      if (name == "dets") {
        dets_name_ = name;
        found_dets = true;
      } else if (name == "labels") {
        labels_name_ = name;
        found_labels = true;
      } else {
        unrecognized_outputs.push_back(name);
      }
    }
  }

  // Positional fallback for engines exported without exact "dets"/"labels"
  // names, assuming (dets, labels) order - matches rf-detr-cpp-orig's
  // binding-discovery convention.
  if (!found_dets && unrecognized_outputs.size() >= 1) {
    dets_name_ = unrecognized_outputs[0];
    found_dets = true;
  }
  if (!found_labels && unrecognized_outputs.size() >= 2) {
    labels_name_ = unrecognized_outputs[1];
    found_labels = true;
  }

  if (!found_input || !found_dets || !found_labels) {
    throw internal::TensorRTException(
      "Failed to find input, dets, or labels tensor in engine "
      "(a segmentation engine with a third 'masks' output is not supported "
      "by this detection-only backend)");
  }
}

void RFDetrTrtBackend::Impl::initialize_memory(const RFDetrTrtBackend::Config & config)
{
  const int num_classes_with_bg = config.num_classes + 1;

  // Calculate memory sizes
  input_size_ = static_cast<size_t>(config.height) * config.width * 3 * sizeof(unsigned char);
  preprocessed_size_ = static_cast<size_t>(3) * config.height * config.width * sizeof(float);
  dets_size_ = static_cast<size_t>(config.num_queries) * 4 * sizeof(float);
  labels_size_ = static_cast<size_t>(config.num_queries) * num_classes_with_bg * sizeof(float);

  // Allocate device memory. If a later allocation throws, everything
  // already assigned above is freed automatically as the exception unwinds
  // through Impl's constructor path - no manual cleanup() bookkeeping.
  buffers_.device_input_raw = cuda_malloc_dev(input_size_);
  buffers_.device_preprocessed = cuda_malloc_dev(preprocessed_size_);
  buffers_.device_dets = cuda_malloc_dev(dets_size_);
  buffers_.device_labels = cuda_malloc_dev(labels_size_);

  // Allocate pinned host memory for reading the two outputs back
  buffers_.pinned_dets = cuda_malloc_host(dets_size_);
  buffers_.pinned_labels = cuda_malloc_host(labels_size_);

  // Bind tensor addresses. Three boolean-returning TensorRT calls (vs.
  // fcn_trt_backend's two) is exactly the repetition TRT_CHECK exists for.
  TRT_CHECK(context_->setTensorAddress(
    input_name_.c_str(), buffers_.device_preprocessed.get()));
  TRT_CHECK(context_->setTensorAddress(
    dets_name_.c_str(), buffers_.device_dets.get()));
  TRT_CHECK(context_->setTensorAddress(
    labels_name_.c_str(), buffers_.device_labels.get()));
}

void RFDetrTrtBackend::Impl::initialize_streams()
{
  cudaStream_t raw = nullptr;
  CUDA_CHECK(cudaStreamCreate(&raw));
  stream_.reset(raw);
}

void RFDetrTrtBackend::Impl::warmup_engine(const RFDetrTrtBackend::Config & config)
{
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input_raw.get(), 0, input_size_, stream_.get()));

  for (int i = 0; i < config.warmup_iterations; ++i) {
    internal::launch_preprocess_kernel(
      static_cast<unsigned char *>(buffers_.device_input_raw.get()),
      static_cast<float *>(buffers_.device_preprocessed.get()),
      config.mean, config.std, config.width, config.height, stream_.get());

    if (!context_->enqueueV3(stream_.get())) {
      throw internal::TensorRTException("Failed to enqueue warmup inference");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
  }

  std::cout << "Engine warmed up with " << config.warmup_iterations << " iterations" << std::endl;
}

std::vector<uint8_t> RFDetrTrtBackend::Impl::load_engine_file(
  const std::string & engine_path) const
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open engine file: " + engine_path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read engine file: " + engine_path);
  }

  return buffer;
}

void RFDetrTrtBackend::Impl::upload_and_preprocess(
  const cv::Mat & image, const RFDetrTrtBackend::Config & config, cudaStream_t stream) const
{
  // Straight H2D copy of the raw uint8 BGR image - no CPU resize/convert
  // step, unlike fcn_trt_backend::Impl::preprocess_image(). image.data is
  // regular (non-pinned) host memory owned by the caller's cv::Mat; a pinned
  // staging buffer would make the copy itself marginally faster, but
  // infer() already synchronizes on this stream before returning, so there
  // is no cross-frame overlap to lose - not worth the extra complexity here.
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_input_raw.get(), image.data,
    input_size_, cudaMemcpyHostToDevice, stream));

  internal::launch_preprocess_kernel(
    static_cast<unsigned char *>(buffers_.device_input_raw.get()),
    static_cast<float *>(buffers_.device_preprocessed.get()),
    config.mean, config.std, config.width, config.height, stream);
}

} // namespace rf_detr_trt_backend
