// local headers
#include "rf_detr_trt_backend/rf_detr_trt_backend.hpp"
#include "internal/rf_detr_trt_backend_impl.hpp"
#include "internal/cuda_check.hpp"


namespace rf_detr_trt_backend
{

RFDetrTrtBackend::RFDetrTrtBackend(const std::string & engine_path, const Config & config)
: config_(config), impl_(std::make_unique<RFDetrTrtBackend::Impl>())
{
  try {
    impl_->initialize(engine_path, config_);
  } catch (const std::exception & e) {
    impl_.reset();
    throw internal::TensorRTException("Initialization failed: " + std::string(e.what()));
  }
}

RFDetrTrtBackend::~RFDetrTrtBackend() = default;

Detections RFDetrTrtBackend::infer(const cv::Mat & image)
{
  std::lock_guard<std::mutex> lock(infer_mutex_);
  return impl_->infer(image, config_);
}

int RFDetrTrtBackend::input_height() const noexcept
{
  return impl_->input_height();
}

int RFDetrTrtBackend::input_width() const noexcept
{
  return impl_->input_width();
}

int RFDetrTrtBackend::num_queries() const noexcept
{
  return impl_->num_queries();
}

int RFDetrTrtBackend::num_classes() const noexcept
{
  return impl_->num_classes();
}

} // namespace rf_detr_trt_backend
