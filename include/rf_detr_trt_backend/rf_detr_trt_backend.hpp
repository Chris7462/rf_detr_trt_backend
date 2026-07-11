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
  struct Config
  {
    int height;
    int width;
    int num_queries;
    int num_classes;
    float mean[3];
    float std[3];
    float score_threshold;
    int warmup_iterations;
    LogLevel log_level;

    Config()
    : height(704), width(704), num_queries(300), num_classes(90),
      mean{0.485f, 0.456f, 0.406f}, std{0.229f, 0.224f, 0.225f},
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

private:
  Config config_;
  class Impl;
  std::unique_ptr<Impl> impl_;
  mutable std::mutex infer_mutex_;
};

} // namespace rf_detr_trt_backend
