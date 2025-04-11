#pragma once

#include "api.h"

#include "wbc_classifier.h"
#include "wbc_localizer.h"

namespace cortex {

class wbc_differential_api final : public api
{
  wbc_localizer localizer_;

  wbc_classifier classifier_;

  std::array<int, 5> counts_{};

  bool finalized_{ false };

  report_header header_;

public:
  void setup() override;

  void teardown() override;

  void reset(const report_header& header) override;

  void update(std::string encoded_img) override;

  void finalize() override;

  [[nodiscard]] auto results() -> nlohmann::json override;
};

} // namespace cortex
