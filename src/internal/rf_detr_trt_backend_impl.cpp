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

RFDetrTrtBackend::Impl::~Impl()
{
  cleanup();
}

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
  upload_and_preprocess(image, config, stream_);

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    throw internal::TensorRTException("Failed to enqueue inference");
  }

  // Copy both outputs back to pinned host memory for CPU-side decode
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_dets, buffers_.device_dets,
    dets_size_, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(buffers_.pinned_labels, buffers_.device_labels,
    labels_size_, cudaMemcpyDeviceToHost, stream_));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_));

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
    buffers_.pinned_dets, buffers_.pinned_labels,
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

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&buffers_.device_input_raw, input_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_preprocessed, preprocessed_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_dets, dets_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_labels, labels_size_));

  // Allocate pinned host memory for reading the two outputs back
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_dets, dets_size_));
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_labels, labels_size_));

  // Bind tensor addresses. Three boolean-returning TensorRT calls (vs.
  // fcn_trt_backend's two) is exactly the repetition TRT_CHECK exists for.
  TRT_CHECK(context_->setTensorAddress(
    input_name_.c_str(), static_cast<void *>(buffers_.device_preprocessed)));
  TRT_CHECK(context_->setTensorAddress(
    dets_name_.c_str(), static_cast<void *>(buffers_.device_dets)));
  TRT_CHECK(context_->setTensorAddress(
    labels_name_.c_str(), static_cast<void *>(buffers_.device_labels)));
}

void RFDetrTrtBackend::Impl::initialize_streams()
{
  CUDA_CHECK(cudaStreamCreate(&stream_));
  if (!stream_) {
    throw internal::TensorRTException("Failed to create CUDA stream");
  }
}

void RFDetrTrtBackend::Impl::warmup_engine(const RFDetrTrtBackend::Config & config)
{
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input_raw, 0, input_size_, stream_));

  for (int i = 0; i < config.warmup_iterations; ++i) {
    internal::launch_preprocess_kernel(
      buffers_.device_input_raw, buffers_.device_preprocessed,
      config.mean, config.std, config.width, config.height, stream_);

    if (!context_->enqueueV3(stream_)) {
      throw internal::TensorRTException("Failed to enqueue warmup inference");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  std::cout << "Engine warmed up with " << config.warmup_iterations << " iterations" << std::endl;
}

void RFDetrTrtBackend::Impl::cleanup() noexcept
{
  // Free pinned host memory
  if (buffers_.pinned_dets) {
    cudaFreeHost(buffers_.pinned_dets);
  }
  if (buffers_.pinned_labels) {
    cudaFreeHost(buffers_.pinned_labels);
  }

  // Free device memory
  if (buffers_.device_input_raw) {
    cudaFree(buffers_.device_input_raw);
  }
  if (buffers_.device_preprocessed) {
    cudaFree(buffers_.device_preprocessed);
  }
  if (buffers_.device_dets) {
    cudaFree(buffers_.device_dets);
  }
  if (buffers_.device_labels) {
    cudaFree(buffers_.device_labels);
  }

  // Reset all pointers to nullptr (good practice)
  buffers_ = MemoryBuffers{};

  // Destroy stream safely
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;  // Mark as destroyed
  }
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
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_input_raw, image.data,
    input_size_, cudaMemcpyHostToDevice, stream));

  internal::launch_preprocess_kernel(
    buffers_.device_input_raw, buffers_.device_preprocessed,
    config.mean, config.std, config.width, config.height, stream);
}

} // namespace rf_detr_trt_backend
