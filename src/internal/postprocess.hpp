#pragma once

#include "rf_detr_trt_backend/rf_detr_trt_backend.hpp"


namespace rf_detr_trt_backend
{

namespace internal
{

// RF-DETR's class head emits `num_classes + 1` logits per query. The leading
// slot (index 0) is the no-object / background class - match upstream
// roboflow/rf-detr Python PostProcess, which excludes it from selection.
inline constexpr int kBackgroundClassIndex = 0;

// Parameters for RF-DETR detection decoding.
struct PostprocessParams
{
  int num_queries{300};          // must match Config::num_queries
  int num_classes_with_bg{91};   // Config::num_classes + 1 (background slot)
  int topk{300};                 // num_select; default = num_queries (Python reference)
  float threshold{0.5f};         // score threshold, matches Config::score_threshold
  int bg_class_index{kBackgroundClassIndex};  // raw class id treated as bg; -1 disables filtering
};

// Map raw class id (0..num_classes_with_bg-1) to dense user-facing class id
// (0..num_classes-1). Shifts down by one when background filtering is
// enabled, so callers never see the no-object slot.
inline int dense_class_from_raw(int raw_class, int bg_class_index) noexcept
{
  return (bg_class_index >= 0) ? (raw_class - 1) : raw_class;
}

// Decode RF-DETR detection outputs (CPU implementation).
//
// Inputs (host pointers, contiguous):
//   dets   : (num_queries, 4) float32, cxcywh normalized to [0,1]
//   labels : (num_queries, num_classes_with_bg) float32, raw logits
//            (sigmoid focal-style head - NOT softmax)
//
// img_w / img_h: dimensions to scale boxes into. Note this is the square
// size RFDetrTrtBackend::infer() actually ran on (Config::width/height) -
// NOT some separate "original image" size like rf-detr-cpp-orig's version
// assumes. Since infer() hard-requires its input already be exactly that
// square, there is no separate original size for the backend to know about;
// un-letterboxing decoded boxes back to the true original image (e.g. a
// KITTI frame) is the caller's responsibility, one layer up.
//
// Output: dense Detections with xyxy boxes in [0, img_w] x [0, img_h]
// pixel coordinates, sorted by descending score.
Detections decode_detections(
  const float * dets, const float * labels, int img_w, int img_h,
  const PostprocessParams & params);

} // namespace internal

} // namespace rf_detr_trt_backend
