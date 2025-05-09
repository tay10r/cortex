#include "focus_tool.h"

#include <implot.h>

#include <stdlib.h>

namespace cortex {

namespace {

class focus_tool_impl final : public focus_tool
{
  int box_size_{};

public:
  focus_tool_impl(const std::vector<std::string>& args)
    : box_size_(atoi(args.at(0).c_str()))
  {
  }

  void render_ui() override
  {
    //
  }

  void render_plot() override
  {
    if (box_size_ <= 0) {
      return;
    }

    if (plot_mouse_box(ImPlot::GetPlotMousePos(), box_size_)) {
      //
    }
  }
};

} // namespace

auto
focus_tool::create(const std::vector<std::string>& args) -> std::unique_ptr<focus_tool>
{
  return std::make_unique<focus_tool_impl>(args);
}

} // namespace cortex
