#include "config.h"

#include <string>

#include <imgui.h>
#include <imgui_stdlib.h>

namespace cortex {

namespace {

class config_widget_impl final : public config_widget
{
  std::string camera_ip_;

  std::string stage_ip_;

public:
  void loop() override
  {
    ImGui::Text("If you have a camera server running on your local network, "
                "enter the IP here.");

    ImGui::InputText("Camera Server", &camera_ip_);

    ImGui::Separator();

    ImGui::Text("If you have a stage server running on your local network, "
                "enter the IP here.");

    ImGui::InputText("Stage Server", &stage_ip_);

    ImGui::Separator();

    ImGui::Text("If you have a stage server running on your local network, "
                "enter the IP here.");

    ImGui::InputText("Database Server", &stage_ip_);
  }
};

} // namespace

auto
config_widget::create() -> std::unique_ptr<config_widget>
{
  return std::make_unique<config_widget_impl>();
}

} // namespace cortex
