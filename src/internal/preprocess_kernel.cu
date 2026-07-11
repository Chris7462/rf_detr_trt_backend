#include "internal/preprocess_kernel.cuh"


namespace rf_detr_trt_backend
{

namespace internal
{

namespace
{

// Direct per-pixel normalize + BGR->RGB + HWC->CHW transform. No resize -
// input is assumed to already be exactly [height, width] (see .cuh).
//
// mean/std are passed as plain kernel arguments (float3), not __constant__
// memory. Unlike fcn_trt_backend's MEAN/STDDEV (fixed compile-time ImageNet
// stats shared by every instance), RFDetrTrtBackend::Config::mean/std are
// per-instance runtime values - e.g. two backends loaded with different
// fine-tuned checkpoints could legitimately have different stats. Using
// __constant__ memory would make that a single process-wide global, which
// two concurrently-alive instances would silently stomp on each other's
// values. Plain kernel args avoid that hazard entirely, at negligible cost
// (a few registers vs. a constant-cache read).
__global__ void preprocess_kernel(
  const unsigned char * __restrict__ src,
  float * __restrict__ dst,
  int width, int height,
  float3 pre_mul, float3 pre_sub)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int pixel_idx = y * width + x;
  const unsigned char * p = src + pixel_idx * 3;

  // OpenCV's HWC layout is BGR: p[0]=B, p[1]=G, p[2]=R. RF-DETR's
  // normalization stats (and the model itself) expect RGB - explicit swap
  // here, unlike relying on an upstream cvtColor call.
  float r = static_cast<float>(p[2]);
  float g = static_cast<float>(p[1]);
  float b = static_cast<float>(p[0]);

  // Normalization is folded on the host (see launch_preprocess_kernel):
  //   nx = (x/255 - mean) / std   =   x * pre_mul - pre_sub
  // trades a per-pixel divide for a per-pixel multiply.
  float nr = r * pre_mul.x - pre_sub.x;
  float ng = g * pre_mul.y - pre_sub.y;
  float nb = b * pre_mul.z - pre_sub.z;

  // Store CHW (channel-first)
  int hw = width * height;
  dst[0 * hw + pixel_idx] = nr;
  dst[1 * hw + pixel_idx] = ng;
  dst[2 * hw + pixel_idx] = nb;
}

} // namespace

void launch_preprocess_kernel(
  const unsigned char * input_data,
  float * output_data,
  const float mean[3],
  const float std[3],
  int width, int height,
  cudaStream_t stream)
{
  constexpr float kInv255 = 1.0f / 255.0f;
  const float3 pre_mul{kInv255 / std[0], kInv255 / std[1], kInv255 / std[2]};
  const float3 pre_sub{mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]};

  dim3 block_size(16, 16);
  dim3 grid_size(
    (width + block_size.x - 1) / block_size.x,
    (height + block_size.y - 1) / block_size.y);

  preprocess_kernel<<<grid_size, block_size, 0, stream>>>(
    input_data, output_data, width, height, pre_mul, pre_sub);
}

} // namespace internal

} // namespace rf_detr_trt_backend
