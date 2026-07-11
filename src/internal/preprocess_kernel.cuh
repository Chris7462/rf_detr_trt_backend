#pragma once

#include <cuda_runtime.h>


namespace rf_detr_trt_backend
{

namespace internal
{

/**
 * @brief GPU accelerated normalize + BGR->RGB + HWC->CHW kernel
 * @details Deliberately does NOT resize, unlike rf-detr-cpp-orig's fused
 * square-resize-normalize kernel: RFDetrTrtBackend::infer() hard-requires
 * the input already be exactly height x width (the ROS2 node owns letterbox
 * padding + resize), so this is a direct per-pixel transform with no
 * interpolation - simpler and cheaper than a fused resize kernel.
 * @param input_data   Input image on GPU: HWC uint8, BGR order (OpenCV
 * native), size [height, width, 3]
 * @param output_data  Output buffer on GPU: CHW float32, RGB order, size
 * [3, height, width]
 * @param mean         Per-channel mean, RGB order
 * @param std          Per-channel standard deviation, RGB order
 * @param width        Image width
 * @param height       Image height
 * @param stream       CUDA stream to launch the kernel on
 */
void launch_preprocess_kernel(
  const unsigned char * input_data,
  float * output_data,
  const float mean[3],
  const float std[3],
  int width, int height,
  cudaStream_t stream);

} // namespace internal

} // namespace rf_detr_trt_backend
