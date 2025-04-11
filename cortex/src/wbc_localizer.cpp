#include "wbc_localizer.h"

#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>

namespace cortex {

void
wbc_localizer::load_model(const std::string& model_filename)
{
  net_ = cv::dnn::readNet(model_filename);
}

auto
wbc_localizer::crop_cells(const cv::Mat& mask, const int crop_size) -> std::vector<cv::Rect2i>
{
  std::vector<cv::Rect2i> crops;

  // 1. Threshold and convert to CV_8U
  cv::Mat binary;
  cv::threshold(mask, binary, 0.5, 255.0, cv::THRESH_BINARY);
  binary.convertTo(binary, CV_8U);

  // 2. Connected components with stats
  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;

  const auto n_labels{ cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S) };

  for (int i = 1; i < n_labels; ++i) { // skip label 0 (background)
    const auto area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area < 50 || area > 5000) {
      continue;
    }

    const auto cx{ centroids.at<double>(i, 0) };
    const auto cy{ centroids.at<double>(i, 1) };

    const auto x0{ static_cast<int>(std::round(cx - static_cast<int>(crop_size / 2))) };
    const auto y0{ static_cast<int>(std::round(cy - static_cast<int>(crop_size / 2))) };

    crops.emplace_back(x0, y0, crop_size, crop_size);
  }

  return crops;
}

auto
wbc_localizer::segment(uint8_t* rgb, const ssize_t w, const ssize_t h) -> cv::Mat
{
  cv::Mat mat(cv::Size(w, h), CV_8UC3, rgb);

  auto blob = cv::dnn::blobFromImage(mat, /*scalefactor=*/1.0 / 255.0, cv::Size(w, h), cv::Scalar(), /*swapRB=*/false);

  net_.setInput(blob);

  auto result = net_.forward();

  std::vector<cv::Mat> outputs;

  cv::dnn::imagesFromBlob(result, outputs);

  cv::Mat output = outputs.at(0);

  return output;
}

auto
wbc_localizer::segment_tiles(uint8_t* rgb, const ssize_t w, const ssize_t h) -> cv::Mat
{
  const ssize_t input_w{ 512 };
  const ssize_t input_h{ 512 };
  const ssize_t output_w{ 64 };
  const ssize_t output_h{ 64 };

  const auto x_stride{ input_w };
  const auto y_stride{ input_h };

  const ssize_t tiles_x{ (w - input_w) / x_stride + 1 };
  const ssize_t tiles_y{ (h - input_h) / y_stride + 1 };

  const ssize_t stitched_w = tiles_x * output_w;
  const ssize_t stitched_h = tiles_y * output_h;

  cv::Mat stitched = cv::Mat::zeros(stitched_h, stitched_w, CV_32F);
  cv::Mat image(cv::Size(w, h), CV_8UC3, rgb);

  for (ssize_t ty = 0; ty < tiles_y; ++ty) {
    for (ssize_t tx = 0; tx < tiles_x; ++tx) {
      const ssize_t x = tx * x_stride;
      const ssize_t y = ty * y_stride;

      cv::Rect roi(x, y, input_w, input_h);
      cv::Mat patch = image(roi);

      // Preprocess and run through model
      const auto scale{ 1.0 / 255.0 };
      cv::Mat blob = cv::dnn::blobFromImage(patch, scale, patch.size(), /*mean=*/cv::Scalar(), /*swapRB=*/true);
      net_.setInput(blob);

      cv::Mat output_blob = net_.forward();
      const ssize_t out_h = output_blob.size[2];
      const ssize_t out_w = output_blob.size[3];
      cv::Mat output(out_h, out_w, CV_32F, output_blob.ptr<float>());

      // Copy into stitched output
      cv::Rect dest_roi(tx * output_w, ty * output_h, output_w, output_h);
      output.copyTo(stitched(dest_roi));
    }
  }

  return stitched;
}

} // namespace cortex
