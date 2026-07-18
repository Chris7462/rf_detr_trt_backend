#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/core.hpp>


namespace rf_detr_trt_backend
{

// Public log level enum, decoupled from nvinfer1::ILogger::Severity.
// Values match the existing "log_level" ROS2 parameter convention
// (0: Internal Error, 1: Error, 2: Warning, 3: Info, 4: Verbose).
enum class LogLevel
{
  kInternalError = 0,
  kError = 1,
  kWarning = 2,
  kInfo = 3,
  kVerbose = 4
};

/**
 * @brief Axis-aligned bounding box in pixel coordinates.
 */
struct Box
{
  float x1{0.0f};
  float y1{0.0f};
  float x2{0.0f};
  float y2{0.0f};

  float width() const noexcept {return x2 - x1;}
  float height() const noexcept {return y2 - y1;}
};

/**
 * @brief A single decoded detection.
 */
struct Detection
{
  Box box;
  int class_id{-1};
  float score{0.0f};
};

using Detections = std::vector<Detection>;

// Optimized TensorRT inference class
class RFDetrTrtBackend
{
public:
  // NOTE: height, width, num_queries, and num_classes are intentionally NOT
  // configurable here. They are properties of the loaded .engine file, not
  // runtime choices - there is no valid scenario where a caller would want
  // RFDetrTrtBackend to use a value different from what the engine actually
  // has. They are resolved from the engine's own tensor shapes at
  // construction time (see find_tensor_names()/initialize_memory() in
  // Impl) and exposed read-only via input_height()/input_width()/
  // num_queries()/num_classes() below, once construction succeeds.
  //
  // mean/std remain configurable because, unlike the fields above, they are
  // not retrievable from the engine itself - they are Python-side constants
  // that existed only at ONNX export time and never made it into the .engine
  // file.
  struct Config
  {
    float mean[3];
    float std[3];
    float score_threshold;
    int warmup_iterations;
    LogLevel log_level;

    Config()
    : mean{0.485f, 0.456f, 0.406f}, std{0.229f, 0.224f, 0.225f},
      score_threshold(0.5f), warmup_iterations(2),
      log_level(LogLevel::kWarning) {}
  };

  explicit RFDetrTrtBackend(const std::string & engine_path, const Config & config = Config());
  ~RFDetrTrtBackend();

  RFDetrTrtBackend(const RFDetrTrtBackend &) = delete;
  RFDetrTrtBackend & operator=(const RFDetrTrtBackend &) = delete;
  RFDetrTrtBackend(RFDetrTrtBackend &&) = delete;
  RFDetrTrtBackend & operator=(RFDetrTrtBackend &&) = delete;

  Detections infer(const cv::Mat & image);

  // Model geometry, resolved from the engine's own tensor shapes at
  // construction time. Only valid to call after construction succeeds
  // (construction throws on failure, so if you have a live instance, these
  // are safe to call).
  int input_height() const noexcept;
  int input_width() const noexcept;
  int num_queries() const noexcept;
  // Dense foreground class count (excludes the background/no-object slot).
  int num_classes() const noexcept;

private:
  Config config_;
  class Impl;
  std::unique_ptr<Impl> impl_;
  mutable std::mutex infer_mutex_;
};

} // namespace rf_detr_trt_backend
