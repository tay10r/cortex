#pragma once

#include <opencv2/dnn.hpp>

#include <vector>

#include <stddef.h>
#include <stdint.h>

namespace cortex {

class wbc_localizer final
{
  cv::dnn::Net net_;

public:
  void load_model(const std::string& model_filename);

  /**
   * @brief Gets the ratio between the input image size and the predicted mask size.
   * */
  [[nodiscard]] auto get_scale_factor() const -> ssize_t { return 8; }

  [[nodiscard]] auto crop_cells(const cv::Mat& mask, const int crop_size) -> std::vector<cv::Rect2i>;

  [[nodiscard]] auto segment(uint8_t* rgb, ssize_t w, ssize_t h) -> cv::Mat;

  [[nodiscard]] auto segment_tiles(uint8_t* rgb, ssize_t w, ssize_t h) -> cv::Mat;
};

} // namespace cortex
