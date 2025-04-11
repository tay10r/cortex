#pragma once

#include <opencv2/dnn.hpp>

#include <utility>

namespace cortex {

enum class wbc_kind
{
  unknown = -1,
  basophils = 0,
  eosinophils = 1,
  lymphocyte = 2,
  monocyte = 3,
  neutrophil = 4,
};

class wbc_classifier final
{
  cv::dnn::Net net_;

public:
  void load_model(const std::string& filename);

  [[nodiscard]] auto classify(uint8_t* rgb, ssize_t w, ssize_t h) -> std::pair<wbc_kind, float>;

  [[nodiscard]] auto classify(cv::Mat& img) -> std::pair<wbc_kind, float>;
};

} // namespace cortex
