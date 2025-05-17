#include "camera_widget.h"

#include "task.h"
#include "visualizer.h"

#include <uikit/shader_compiler.hpp>

#include <imgui.h>
#include <imgui_stdlib.h>

#include <implot.h>

#include <TextEditor.h>

#include <GLES2/gl2.h>

namespace cortex {

namespace {

class camera_widget_impl final : public camera_widget
{
  std::string camera_server_{ "127.0.0.1" };

  std::unique_ptr<visualizer> visualizer_{ visualizer::create() };

  std::unique_ptr<task> frame_task_;

  std::unique_ptr<task> config_get_task_;

  std::unique_ptr<task> config_set_task_;

  bool errored_{ false };

  int exposure_{ 1000 };

  float gain_{ 1 };

  int light_level_{ 127 };

public:
  void setup() override
  {
    visualizer_->setup();
    //
  }

  void teardown() override { visualizer_->teardown(); }

  void render() override
  {
    if (ImGui::Button("Save")) {
      save_frame();
    }

    ImGui::SameLine();

    ImGui::BeginDisabled(!!frame_task_);

    if (ImGui::Button("Request Frame")) {
      request_frame();
    }

    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGui::SetNextItemWidth(ImGui::CalcTextSize("000.000.000.000 IPv4").x);

    ImGui::InputText("IPv4", &camera_server_);

    if (frame_task_) {
      ImGui::ProgressBar(frame_task_->progress());
    }

    if (errored_) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
      ImGui::TextUnformatted("Failed to fetch frame from server.");
      ImGui::PopStyleColor();
    }

    if (ImGui::CollapsingHeader("Controls")) {

      ImGui::SliderInt("Light Duty Cycle", &light_level_, 0, 255);

      ImGui::SliderInt("Exposure [microseconds]", &exposure_, 0, 1'000'000, nullptr);

      ImGui::SliderFloat("Gain", &gain_, 1, 8);

      ImGui::BeginDisabled(!!config_get_task_);

      if (ImGui::Button("Get Config")) {
        config_get_task_ = task::http_get(config_url());
      }

      ImGui::EndDisabled();

      ImGui::SameLine();

      ImGui::BeginDisabled(!!config_set_task_);

      if (ImGui::Button("Set Config")) {
        config_set_task_ = task::http_put(config_url(), pack_config());
      }

      ImGui::EndDisabled();
    }

    visualizer_->loop();
  }

  void poll() override
  {
    if (frame_task_) {
      poll_frame_task();
    }

    if (config_get_task_) {
      poll_config_get_task();
    }

    if (config_set_task_) {
      poll_config_set_task();
    }
  }

protected:
  [[nodiscard]] auto pack_config() -> std::string
  {
    std::string buffer;
    buffer.resize(8);
    *reinterpret_cast<int*>(buffer.data()) = exposure_;
    *reinterpret_cast<float*>(buffer.data() + 4) = gain_;
    return buffer;
  }

  void poll_frame_task()
  {
    frame_task_->poll();

    if (frame_task_->done()) {

      if (frame_task_->failed()) {
        errored_ = true;
      } else {
        handle_frame();
      }

      frame_task_.reset();
    }
  }

  void poll_config_get_task()
  {
    config_get_task_->poll();

    if (config_get_task_->done()) {

      if (!config_get_task_->failed()) {
        handle_config(config_get_task_->data(), config_get_task_->size());
      }

      config_get_task_.reset();
    }
  }

  void poll_config_set_task()
  {
    config_set_task_->poll();

    if (config_set_task_->done()) {

      if (!config_set_task_->failed()) {
        errored_ = "failed to set config";
      }

      config_set_task_.reset();
    }
  }

  void handle_frame()
  {
    const auto w{ frame_task_->get_header_long("x-image-width",
                                               /*fallback=*/0L) };

    const auto h{ frame_task_->get_header_long("x-image-height",
                                               /*fallback=*/0L) };

    if ((w <= 0) || (h <= 0) || ((static_cast<size_t>(w) * static_cast<size_t>(h) * 2) != frame_task_->size())) {
      return;
    }

    visualizer_->update(frame_task_->data(), w, h);
  }

  void handle_config(const void* data, const size_t size)
  {
    static_assert(sizeof(int) == 4, "Size of int must be 4 bytes.");
    static_assert(sizeof(float) == 4, "Size of float must be 4 bytes.");
    if (size == 8) {
      exposure_ = reinterpret_cast<const int*>(data)[0];
      gain_ = reinterpret_cast<const float*>(data)[1];
    }
  }

  void request_frame()
  {
    if (frame_task_) {
      return;
    }

    const std::string url{ std::string("http://") + camera_server_ +
                           ":6500/snapshot?light_duty_cycle=" + std::to_string(light_level_) };

    frame_task_ = task::http_get(url);

    errored_ = false;
  }

  void save_frame()
  {
    //
  }

  [[nodiscard]] auto config_url() const -> std::string
  {
    std::string url;
    url += "http://";
    url += camera_server_;
    url += ":6500/config";
    return url;
  }
};

} // namespace

auto
camera_widget::create() -> std::unique_ptr<camera_widget>
{
  return std::make_unique<camera_widget_impl>();
}

} // namespace cortex
