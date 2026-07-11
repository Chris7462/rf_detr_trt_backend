#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <stdexcept>

// CUDA includes
#include <cuda_runtime.h>


namespace rf_detr_trt_backend
{

namespace internal
{

// Custom exception classes
class TensorRTException : public std::runtime_error
{
public:
  explicit TensorRTException(const std::string & message)
  : std::runtime_error("TensorRT Error: " + message) {}
};

class CudaException : public std::runtime_error
{
public:
  explicit CudaException(const std::string & message, cudaError_t error)
  : std::runtime_error("CUDA Error: " + message + " (" + cudaGetErrorString(error) + ")") {}
};

} // namespace internal

} // namespace rf_detr_trt_backend

// CUDA error checking macro
// NOTE: Preprocessor macros are not namespace-scoped, so this expands to
// rf_detr_trt_backend::internal::CudaException regardless of the namespace
// context it's invoked from. Kept outside the namespace blocks above since
// macro definitions conventionally aren't nested.
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      throw rf_detr_trt_backend::internal::CudaException(#call, error); \
    } \
  } while(0)

// TensorRT boolean-API error checking macro.
// Unlike fcn_trt_backend (which only has one or two boolean-returning TRT
// calls and throws inline at each), rf_detr_trt_backend has three
// setTensorAddress() calls (input, dets, labels) plus other boolean checks
// around binding discovery - enough repetition to warrant a macro, so error
// messages stay consistent across all call sites instead of depending on
// each one being written carefully by hand.
#define TRT_CHECK(expr) \
  do { \
    if (!(expr)) { \
      throw rf_detr_trt_backend::internal::TensorRTException(#expr); \
    } \
  } while(0)
