#pragma once

#include <memory>
#include <string>
#include <vector>

#include <implot.h>

namespace cortex {

class tool
{
public:
  static auto create(const std::string& type, const std::vector<std::string>& args) -> std::unique_ptr<tool>;

  virtual ~tool() = default;

  virtual void render_ui() = 0;

  virtual void render_plot() = 0;

protected:
  static bool plot_mouse_box(const ImPlotPoint& p, const int box_size);
};

} // namespace cortex
