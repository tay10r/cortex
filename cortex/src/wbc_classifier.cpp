#include "wbc_classifier.h"

namespace cortex {

void
wbc_classifier::load_model(const std::string& filename)
{
  net_ = cv::dnn::readNet(filename);
}

auto
wbc_classifier::classify(cv::Mat& input) -> std::pair<wbc_kind, float>
{
  auto blob = cv::dnn::blobFromImage(input, /*scalefactor=*/1.0 / 255.0, input.size(), cv::Scalar(), /*swapRB=*/true);

  net_.setInput(blob);

  const auto result = net_.forward();

  cv::Point class_id_point;

  auto confidence{ 0.0 };

  cv::minMaxLoc(result.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);

  const auto class_id = class_id_point.x;

  return { static_cast<wbc_kind>(class_id), static_cast<float>(confidence) };
}

auto
wbc_classifier::classify(uint8_t* rgb, ssize_t w, ssize_t h) -> std::pair<wbc_kind, float>
{
  cv::Mat mat(cv::Size(w, h), CV_8UC3, rgb);

  return classify(mat);
}

} // namespace cortex
