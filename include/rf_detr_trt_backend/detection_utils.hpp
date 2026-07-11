#pragma once

// C++ standard library includes
#include <functional>

// OpenCV includes
#include <opencv2/core.hpp>

// local header
#include "rf_detr_trt_backend/rf_detr_trt_backend.hpp"


namespace rf_detr_trt_backend
{

namespace utils
{
/**
 * @brief Function mapping a class id to a display name, e.g. from a
 * fine-tuned KITTI class map (0 -> "Car", 1 -> "Pedestrian", ...).
 * @details std::function (rather than a raw function pointer) so callers
 * can pass a capturing lambda, e.g. one that looks up names in a
 * config-loaded std::unordered_map<int, std::string>.
 */
using LabelForClassFn = std::function<const char * (int)>;

/**
 * @brief COCO 91-id sparse class labels. Index 0 is the background
 * "no-object" class emitted by the DETR-style class head. Indices 12, 26,
 * 29, 30, 45, 66, 68, 69, 71, 83 are reserved gaps in the COCO 91-id
 * mapping and are never produced by the trained model - kept as
 * placeholders so the array stays index-aligned.
 */
inline constexpr const char * kCocoLabels91[91] = {
  "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
  "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "__unused_12__", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
  "__unused_26__", "backpack", "umbrella", "__unused_29__", "__unused_30__",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "__unused_45__", "wine glass",
  "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
  "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
  "chair", "couch", "potted plant", "bed", "__unused_66__", "dining table",
  "__unused_68__", "__unused_69__", "toilet", "__unused_71__", "tv",
  "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
  "oven", "toaster", "sink", "refrigerator", "__unused_83__", "book",
  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
};

/**
 * @brief Human-readable COCO class name for a dense foreground class id.
 * @details `dense_id` is RFDetrTrtBackend::infer()'s Detection::class_id
 * (range [0, 89]) - already shifted down by one from the raw 91-slot logit
 * index to exclude the background slot. This re-applies that +1 offset to
 * index back into kCocoLabels91.
 * @param dense_id A Detection::class_id value.
 * @return The class name, or "?" if dense_id is out of the valid [0, 89] range.
 */
inline const char * coco_label(int dense_id) noexcept
{
  if (dense_id < 0 || dense_id + 1 >= 91) {
    return "?";
  }
  return kCocoLabels91[dense_id + 1];
}

/**
 * @brief Function mapping a class id to a BGR box/label color.
 * @details std::function, same rationale as LabelForClassFn. Optional - if
 * not provided, draw_detections() auto-generates a distinct color per
 * class_id (golden-angle hue rotation) rather than assuming any particular
 * dataset's classes, so it works unchanged whether num_classes is 3 (a
 * KITTI fine-tune) or 90 (stock COCO).
 */
using ColorForClassFn = std::function<cv::Scalar (int)>;

/**
 * @brief Draw detections onto a copy of an image: box outline + "<label> <score>" text.
 * @details Does not modify `image` - matches create_overlay()'s non-mutating
 * convention, since callers may hold a shared reference to `image` (e.g. a
 * cv_bridge::toCvShare() buffer backed by the incoming ROS message).
 * @param image Source image. Must be CV_8UC3.
 * @param detections Detections to draw, with boxes in `image`'s pixel coordinates.
 * @param label_for_class Optional class id -> display name lookup. If empty
 * (default), falls back to coco_label() - COCO class names. Pass your own
 * lookup to override (e.g. a custom class map, if that ever changes).
 * @param color_for_class Optional class id -> BGR color lookup. If empty
 * (default), a distinct color is auto-generated per class_id.
 * @return A new image, cloned from `image` with detections drawn onto it.
 */
cv::Mat draw_detections(
  const cv::Mat & image, const Detections & detections,
  const LabelForClassFn & label_for_class = nullptr,
  const ColorForClassFn & color_for_class = nullptr);

/**
 * @brief Transform parameters produced by letterbox_square(), needed to map
 * detected boxes (in the padded square's pixel space) back to the original
 * image's coordinates via unletterbox_detections():
 *   orig = (padded - pad) / scale
 */
struct LetterboxInfo
{
  float scale{1.0f};
  int pad_x{0};
  int pad_y{0};
  int orig_width{0};
  int orig_height{0};
};

/**
 * @brief Resize preserving aspect ratio, then center-pad to a square.
 * @details Resize-then-pad, not pad-then-resize: resizing first keeps the
 * CUDA/CPU work proportional to the (usually smaller) source image rather
 * than a full target_size x target_size intermediate buffer. If `image` is
 * already square, this collapses to a plain resize with zero padding -
 * mathematically identical to the general case, just without an unnecessary
 * canvas allocation + copy.
 * @param image Source image.
 * @param target_size Output side length (output is always exactly
 * target_size x target_size, regardless of input aspect ratio - this is
 * what allows RFDetrTrtBackend::infer() to safely hard-require a fixed
 * square size).
 * @param info Optional out-parameter; if non-null, filled with everything
 * unletterbox_detections() needs to map detections back to `image`'s
 * original coordinates.
 * @return A new target_size x target_size image. Padding is black.
 */
cv::Mat letterbox_square(
  const cv::Mat & image, int target_size, LetterboxInfo * info = nullptr);

/**
 * @brief Map detections from the padded square's pixel space (what
 * RFDetrTrtBackend::infer() returns) back to the original image's pixel
 * space, undoing letterbox_square()'s transform.
 * @details Applies `orig = (padded - pad) / scale` to each box, then clips
 * to [0, orig_width] x [0, orig_height]. Degenerate (zero-area) boxes after
 * clipping are dropped.
 * @param detections Detections in the padded square's coordinate space.
 * @param info The LetterboxInfo produced by the letterbox_square() call
 * that generated the image these detections came from.
 * @return A new Detections with boxes in the original image's coordinates.
 * class_id and score are unchanged.
 */
Detections unletterbox_detections(
  const Detections & detections, const LetterboxInfo & info);

} // namespace utils

} // namespace rf_detr_trt_backend
