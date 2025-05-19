#pragma once

#include "tool.h"

#include "white_balance.h"

#include <stdint.h>

namespace cortex {

class white_balance_tool final : public tool
{
  int box_size_{ 64 };

  float quantile_{ 0.95F };

  histogram hist_;

  float balance_[3]{ 1.0F, 1.0F, 1.0F };

public:
  void render_plot(const image& img, const long x, const long y) override;

  void render_ui() override;

  void on_select() override;

  void on_deselect() override;

protected:
  void render_plot();
};

} // namespace cortex
