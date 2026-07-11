// C++ standard library includes
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local header
#include "rf_detr_trt_backend/detection_utils.hpp"


namespace rf_detr_trt_backend
{

namespace utils
{

namespace
{

// Deterministic, evenly-distributed distinct color per class_id, with no
// hardcoded per-class table - unlike fcos_trt_backend's 90-entry COCO color
// map, this works unchanged whether num_classes is 3 (a KITTI fine-tune) or
// 90 (stock COCO). Golden-angle hue rotation is a standard technique for
// generating N maximally-separated hues without knowing N in advance.
cv::Scalar default_color_for_class(int class_id)
{
  constexpr float kGoldenAngle = 137.508f;
  constexpr float kSaturation = 0.85f;
  constexpr float kValue = 0.95f;

  float hue = std::fmod(static_cast<float>(class_id) * kGoldenAngle, 360.0f);
  if (hue < 0.0f) {
    hue += 360.0f;  // defensive: class_id is expected >= 0, but don't assume
  }

  // Standard HSV -> RGB conversion.
  const float c = kValue * kSaturation;
  const float x = c * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
  const float m = kValue - c;

  float r, g, b;
  if (hue < 60.0f) {
    r = c; g = x; b = 0.0f;
  } else if (hue < 120.0f) {
    r = x; g = c; b = 0.0f;
  } else if (hue < 180.0f) {
    r = 0.0f; g = c; b = x;
  } else if (hue < 240.0f) {
    r = 0.0f; g = x; b = c;
  } else if (hue < 300.0f) {
    r = x; g = 0.0f; b = c;
  } else {
    r = c; g = 0.0f; b = x;
  }

  return cv::Scalar(
    static_cast<int>((b + m) * 255.0f),
    static_cast<int>((g + m) * 255.0f),
    static_cast<int>((r + m) * 255.0f));  // BGR order for OpenCV
}

// White text on dark backgrounds, black text on light backgrounds - needed
// now that box_color varies per class (auto-generated colors span both
// light and dark hues), unlike the old single fixed green where a constant
// black text color was always legible.
cv::Scalar text_color_for_background(const cv::Scalar & bg)
{
  const double brightness = (bg[0] + bg[1] + bg[2]) / 3.0;
  return brightness < 128.0 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);
}

std::string format_label(
  const Detection & det, const LabelForClassFn & label_for_class)
{
  std::string name;
  if (label_for_class) {
    const char * label = label_for_class(det.class_id);
    name = label ? label : coco_label(det.class_id);
  } else {
    name = coco_label(det.class_id);
  }

  char score_buf[8];
  std::snprintf(score_buf, sizeof(score_buf), "%.2f", det.score);

  return name + " " + score_buf;
}

} // namespace

cv::Mat draw_detections(
  const cv::Mat & image, const Detections & detections,
  const LabelForClassFn & label_for_class,
  const ColorForClassFn & color_for_class)
{
  // Clone up front - `image` is never modified, matching create_overlay()'s
  // convention. Safe to call on a cv_bridge::toCvShare() buffer.
  cv::Mat result = image.clone();

  constexpr int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  constexpr double kFontScale = 0.5;
  constexpr int kTextThickness = 1;
  constexpr int kBoxThickness = 2;

  for (const auto & det : detections) {
    const cv::Scalar box_color = color_for_class ?
      color_for_class(det.class_id) : default_color_for_class(det.class_id);
    const cv::Scalar text_color = text_color_for_background(box_color);

    const cv::Point p1(
      static_cast<int>(det.box.x1), static_cast<int>(det.box.y1));
    const cv::Point p2(
      static_cast<int>(det.box.x2), static_cast<int>(det.box.y2));

    cv::rectangle(result, p1, p2, box_color, kBoxThickness);

    const std::string label_text = format_label(det, label_for_class);

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
      label_text, kFontFace, kFontScale, kTextThickness, &baseline);

    // Filled background behind the label text, sitting just above the box's
    // top-left corner (clamped so it never draws above the image top).
    const int label_top = std::max(0, p1.y - text_size.height - 4);
    cv::rectangle(
      result,
      cv::Point(p1.x, label_top),
      cv::Point(p1.x + text_size.width, label_top + text_size.height + 4),
      box_color, cv::FILLED);

    cv::putText(
      result, label_text, cv::Point(p1.x, label_top + text_size.height + 1),
      kFontFace, kFontScale, text_color, kTextThickness);
  }

  return result;
}

cv::Mat letterbox_square(
  const cv::Mat & image, int target_size, LetterboxInfo * info)
{
  const int w = image.cols;
  const int h = image.rows;

  // Already square: the general path below would collapse to this anyway
  // (scale_x == scale_y, zero padding) - skip the canvas allocation + copy.
  if (w == h) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_size, target_size));
    if (info) {
      info->scale = static_cast<float>(target_size) / w;
      info->pad_x = 0;
      info->pad_y = 0;
      info->orig_width = w;
      info->orig_height = h;
    }
    return resized;
  }

  const float scale = std::min(
    static_cast<float>(target_size) / w, static_cast<float>(target_size) / h);
  const int new_w = static_cast<int>(std::round(w * scale));
  const int new_h = static_cast<int>(std::round(h * scale));

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(new_w, new_h));

  cv::Mat canvas(target_size, target_size, image.type(), cv::Scalar(0, 0, 0));
  const int pad_x = (target_size - new_w) / 2;
  const int pad_y = (target_size - new_h) / 2;
  resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

  if (info) {
    info->scale = scale;
    info->pad_x = pad_x;
    info->pad_y = pad_y;
    info->orig_width = w;
    info->orig_height = h;
  }

  return canvas;
}

Detections unletterbox_detections(
  const Detections & detections, const LetterboxInfo & info)
{
  Detections out;
  out.reserve(detections.size());

  const float ow = static_cast<float>(info.orig_width);
  const float oh = static_cast<float>(info.orig_height);

  for (const auto & d : detections) {
    Detection nd = d;  // copies class_id and score unchanged

    nd.box.x1 = (d.box.x1 - info.pad_x) / info.scale;
    nd.box.y1 = (d.box.y1 - info.pad_y) / info.scale;
    nd.box.x2 = (d.box.x2 - info.pad_x) / info.scale;
    nd.box.y2 = (d.box.y2 - info.pad_y) / info.scale;

    nd.box.x1 = std::clamp(nd.box.x1, 0.0f, ow);
    nd.box.y1 = std::clamp(nd.box.y1, 0.0f, oh);
    nd.box.x2 = std::clamp(nd.box.x2, 0.0f, ow);
    nd.box.y2 = std::clamp(nd.box.y2, 0.0f, oh);

    // A box entirely within the letterbox padding maps to zero area after
    // clipping - drop it rather than emit a degenerate detection.
    if (nd.box.width() > 0.0f && nd.box.height() > 0.0f) {
      out.push_back(std::move(nd));
    }
  }

  return out;
}

} // namespace utils

} // namespace rf_detr_trt_backend
