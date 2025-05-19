#include "tool.h"

#include "image.h"

#include <implot.h>

namespace cortex {

auto
tool::plot_mouse_box(const image& img, const long x, const long y, const long box_size) -> bool
{
  const auto s{ static_cast<double>(box_size >> 1) };

  const auto tx = static_cast<double>(x);
  const auto ty = static_cast<double>(img.height() - (y + 1));

  const double xl[4]{ tx - s, tx + s, tx + s, tx - s };
  const double yl[4]{ ty - s, ty - s, ty + s, ty + s };

  const auto is_valid{ is_box_valid(img, x, y, box_size) };

  if (ImPlot::IsPlotHovered()) {

    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 4);

    if (is_valid) {
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0, 1, 0, 1));
    } else {
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 0, 0, 1));
    }

    ImPlot::PlotLine("##mousebox", xl, yl, 4, ImPlotLineFlags_Loop);

    ImPlot::PopStyleVar();

    ImPlot::PopStyleColor();

    if (is_valid && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
      return true;
    }
  }

  return false;
}

auto
tool::is_box_valid(const image& img, const long x, const long y, const long box_size) -> bool
{
  if (box_size < 0) {
    return false;
  }

  const auto r{ box_size / 2 };

  if (((x - r) < 0) || ((y - r) < 0)) {
    return false;
  }

  const auto w{ static_cast<long>(img.width()) };
  const auto h{ static_cast<long>(img.height()) };

  if (((x + r) > w) || ((y + r) > h)) {
    return false;
  }

  return true;
}

auto
tool::crop_box(const image& img, const long x, const long y, const long box_size) -> image_view
{
  if (!is_box_valid(img, x, y, box_size)) {
    return image_view();
  }

  const auto r{ box_size / 2 };

  return img.view(x - r, y - r, box_size, box_size);
}

} // namespace cortex
