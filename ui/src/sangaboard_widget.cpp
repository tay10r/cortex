#include "sangaboard_widget.h"

#include "task.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <sstream>
#include <string>

namespace cortex {

namespace {

class sangaboard_widget_impl final : public sangaboard_widget
{
  std::string server_ip_{ "127.0.0.1" };

  int x_{};

  int y_{};

  int z_{};

  float illumination_{ 1.0F };

  std::unique_ptr<task> command_task_;

  bool failed_{ false };

  std::string response_;

public:
  void setup() override
  {
    //
  }

  void teardown() override
  {
    //
  }

  void render() override
  {
    const auto disabled{ !!command_task_ };

    ImGui::InputText("IPv4", &server_ip_);

    ImGui::DragInt("##Move X", &x_);
    ImGui::SameLine();
    ImGui::BeginDisabled(disabled);
    if (ImGui::Button("Move X")) {
      send_command("mrx", x_);
      x_ = 0;
    }
    ImGui::EndDisabled();

    ImGui::DragInt("##Move Y", &y_);
    ImGui::SameLine();
    ImGui::BeginDisabled(disabled);
    if (ImGui::Button("Move Y")) {
      send_command("mry", y_);
      y_ = 0;
    }
    ImGui::EndDisabled();

    ImGui::DragInt("##Move Z", &z_);
    ImGui::SameLine();
    ImGui::BeginDisabled(disabled);
    if (ImGui::Button("Move Z")) {
      send_command("mrz", z_);
      z_ = 0;
    }
    ImGui::EndDisabled();

    ImGui::SliderFloat("##Illumination", &illumination_, 0, 1);
    ImGui::SameLine();
    if (ImGui::Button("Change Illumination")) {
      send_command("led_cc", illumination_);
    }

    if (failed_) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
      ImGui::TextUnformatted("Failed to send command.");
      ImGui::PopStyleColor();
    } else if (!response_.empty()) {
      ImGui::Text("Response: %s\n", response_.c_str());
    }
  }

  void poll() override
  {
    if (!command_task_) {
      return;
    }

    command_task_->poll();

    if (command_task_->done()) {

      if (command_task_->failed()) {
        failed_ = true;
      } else {
        response_ = std::string(static_cast<const char*>(command_task_->data()), command_task_->size());
      }

      command_task_.reset();
    }
  }

protected:
  [[nodiscard]] auto get_url() -> std::string
  {
    std::ostringstream stream;
    stream << "http://" << server_ip_ << ":6400/command";
    return stream.str();
  }

  template<typename Arg>
  void send_command(const char* command, const Arg& arg)
  {
    std::ostringstream stream;
    stream << command;
    stream << ' ';
    stream << arg;
    stream << '\n';
    command_task_ = task::http_post(get_url(), stream.str());
    failed_ = false;
    response_.clear();
  }
};

} // namespace

auto
sangaboard_widget::create() -> std::unique_ptr<sangaboard_widget>
{
  return std::make_unique<sangaboard_widget_impl>();
}

} // namespace cortex
