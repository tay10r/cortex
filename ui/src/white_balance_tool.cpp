#include "white_balance_tool.h"

#include "image.h"

#include <imgui.h>
#include <implot.h>

namespace cortex {

void
white_balance_tool::render_plot(const image& img, const long x, const long y)
{
  if (plot_mouse_box(img, x, y, box_size_)) {

    hist_.reset();

    compute_histogram(tool::crop_box(img, x, y, box_size_), hist_);

    float rgb[3]{};

    hist_.compute_quantile_values(quantile_, rgb);

    balance_[0] = rgb[1] / rgb[0];
    balance_[1] = 1.0F;
    balance_[2] = rgb[1] / rgb[2];
  }
}

void
white_balance_tool::render_ui()
{
  ImGui::SliderInt("Box Size [pixels]", &box_size_, 1, 256);

  ImGui::SliderFloat("Quantile", &quantile_, 0, 1);

  ImGui::InputFloat3("Balance", balance_, nullptr, ImGuiInputTextFlags_ReadOnly);

  render_plot();
}

void
white_balance_tool::on_select()
{
  //
}

void
white_balance_tool::on_deselect()
{
  //
}

void
white_balance_tool::render_plot()
{
  if (!ImPlot::BeginPlot("Histogram", ImVec2(-1, 0), ImPlotFlags_NoFrame)) {
    return;
  }

  ImPlot::SetupAxes("Bin", "Count", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 0, 0, 1));
  ImPlot::PlotLine("R", hist_.r, 256);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0, 1, 0, 1));
  ImPlot::PlotLine("G", hist_.g, 256);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0, 0, 1, 1));
  ImPlot::PlotLine("B", hist_.b, 256);

  ImPlot::PopStyleColor(3);

  ImPlot::EndPlot();
}

} // namespace cortex
