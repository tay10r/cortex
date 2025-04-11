#include "wbc_differential_api.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

namespace cortex {

namespace {

[[nodiscard]] auto
crop_with_padding(const cv::Mat& image, const cv::Rect& rect) -> cv::Mat
{
  // Create black canvas of requested crop size
  cv::Mat result(rect.height, rect.width, image.type(), cv::Scalar::all(0));

  // Compute the intersection of the requested rect and the image bounds
  cv::Rect image_bounds(0, 0, image.cols, image.rows);
  cv::Rect src_roi = rect & image_bounds;

  if (src_roi.empty()) {
    return result; // Entire rect is out-of-bounds, return all black
  }

  // Compute destination position inside the padded result
  const auto dst_x{ std::max(0, src_roi.x - rect.x) }; // offset from left
  const auto dst_y{ std::max(0, src_roi.y - rect.y) }; // offset from top

  // Copy the in-bounds region into the result at the correct offset
  cv::Mat dst_roi = result(cv::Rect(dst_x, dst_y, src_roi.width, src_roi.height));
  image(src_roi).copyTo(dst_roi);

  return result;
}
} // namespace

void
wbc_differential_api::setup()
{
  classifier_.load_model("WBC_Classifier-v1.0.onnx");
  localizer_.load_model("WBC_Localizer-v1.0.onnx");
}

void
wbc_differential_api::teardown()
{
  //
}

void
wbc_differential_api::reset(const report_header& header)
{
  header_ = header;

  for (auto& c : counts_) {
    c = 0;
  }

  finalized_ = false;
}

void
wbc_differential_api::update(std::string encoded_img)
{
  if (finalized_) {
    return;
  }
  cv::Mat rawData(1, static_cast<int>(encoded_img.size()), CV_8UC1, const_cast<char*>(encoded_img.data()));

  cv::Mat image = cv::imdecode(rawData, cv::IMREAD_COLOR);

  const auto size = image.size();

  const auto mask = localizer_.segment_tiles(image.data, size.width, size.height);

  const auto scale = localizer_.get_scale_factor();

  const auto crop_size{ 256 };

  const auto cell_boxes = localizer_.crop_cells(mask, /*crop_size=*/crop_size / scale);

  for (const auto& box : cell_boxes) {

    const cv::Rect scaled_box{ static_cast<int>(box.x * scale), static_cast<int>(box.y * scale), crop_size, crop_size };

    auto cell_chip = crop_with_padding(image, scaled_box);

    // TODO : find a way to produce this information that's not so hacky
    // std::ostringstream stream;
    // stream << "chip_" << i << ".png";
    // cv::imwrite(stream.str(), cell_chip);
    // i++;

    const auto [cell_class, confidence] = classifier_.classify(cell_chip);

    counts_.at(static_cast<int>(cell_class))++;
  }
}

void
wbc_differential_api::finalize()
{
  finalized_ = true;
}

auto
wbc_differential_api::results() -> nlohmann::json
{
  nlohmann::json counts;
  counts["Basophils"] = counts_[0];
  counts["Eosinophils"] = counts_[1];
  counts["Lymphocytes"] = counts_[2];
  counts["Monocytes"] = counts_[3];
  counts["Neutrophils"] = counts_[4];

  const auto total = static_cast<float>(counts_[0] + counts_[1] + counts_[2] + counts_[3] + counts_[4]);

  nlohmann::json distribution;

  if (total > 0) {
    distribution["Basophils"] = counts_[0] / total;
    distribution["Eosinophils"] = counts_[1] / total;
    distribution["Lymphocytes"] = counts_[2] / total;
    distribution["Monocytes"] = counts_[3] / total;
    distribution["Neutrophils"] = counts_[4] / total;
  } else {
    distribution["Basophils"] = 0.0;
    distribution["Eosinophils"] = 0.0;
    distribution["Lymphocytes"] = 0.0;
    distribution["Monocytes"] = 0.0;
    distribution["Neutrophils"] = 0.0;
  }

  nlohmann::json result;
  result["report_id"] = header_.id;
  result["timestamp"] = header_.timestamp;
  result["notes"] = header_.notes;
  result["counts"] = counts;
  result["distribution"] = distribution;
  result["complete"] = finalized_;

  return result;
}

} // namespace cortex
