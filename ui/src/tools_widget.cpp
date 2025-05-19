#include "tools_widget.h"

#include "image.h"
#include "tool.h"
#include "white_balance_tool.h"

#include <imgui.h>
#include <implot.h>

#include <string>
#include <vector>

namespace cortex {

namespace {

class tools_widget_impl final : public tools_widget
{
  size_t selected_{ ~static_cast<size_t>(0) };

  std::vector<std::unique_ptr<tool>> tools_;

  std::vector<std::string> tool_names_;

public:
  void setup() override { add_tool("White Balance", std::make_unique<white_balance_tool>()); }

  void teardown() override
  {
    //
  }

  void render() override
  {
    for (size_t i = 0; i < tool_names_.size(); i++) {
      if (ImGui::Selectable(tool_names_[i].c_str(), i == selected_)) {
        auto* t = get_selected();
        if (t) {
          t->on_deselect();
        }
        selected_ = i;
        get_selected()->on_select();
      }
    }

    ImGui::Separator();

    auto* t = get_selected();
    if (t) {
      t->render_ui();
    }
  }

  void poll() override
  {
    //
  }

  void plot(const image& img) override
  {
    if (!ImPlot::IsPlotHovered()) {
      return;
    }

    auto* t = get_selected();
    if (t) {
      const auto p = ImPlot::GetPlotMousePos();
      const auto x = static_cast<long>(p.x);
      const auto y = static_cast<long>(img.height()) - (static_cast<long>(p.y) + 1);

      t->render_plot(img, x, y);
    }
  }

protected:
  [[nodiscard]] auto get_selected() -> tool*
  {
    if (selected_ < tools_.size()) {
      return tools_.at(selected_).get();
    } else {
      return nullptr;
    }
  }

  void add_tool(std::string name, std::unique_ptr<tool> t)
  {
    tool_names_.emplace_back(std::move(name));

    tools_.emplace_back(std::move(t));
  }
};

} // namespace

auto
tools_widget::create() -> std::unique_ptr<tools_widget>
{
  return std::make_unique<tools_widget_impl>();
}

} // namespace cortex
