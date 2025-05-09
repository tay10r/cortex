#include "tool.h"

#include "tools/focus_tool.h"

namespace cortex {

namespace {

class tool_impl final : public tool
{
public:
};

} // namespace

auto
tool::create(const std::string& type, const std::vector<std::string>& args) -> std::unique_ptr<tool>
{
  if (type == "focus") {
    return focus_tool::create(args);
  }

  return nullptr;
}

bool
tool::plot_mouse_box(const ImPlotPoint& p, const int box_size)
{
  const auto s{ static_cast<double>(box_size) };

  const double x[4]{ p.x, p.x + box_size, p.x + box_size, p.x };

  const double y[4]{ p.y, p.y, p.y + s, p.y + s };

  if (ImPlot::IsPlotHovered()) {

    ImPlot::PlotLine("##mousebox", x, y, 4, ImPlotLineFlags_Loop);

    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      return true;
    }
  }

  return false;
}

} // namespace cortex
