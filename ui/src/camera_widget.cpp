#include "camera_widget.h"

#include "transfer.h"
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

  std::unique_ptr<transfer> frame_transfer_;

  bool errored_{ false };

public:
  void setup() override { visualizer_->setup(); }

  void teardown() override { visualizer_->teardown(); }

  void render() override
  {
    if (ImGui::Button("Save")) {
      save_frame();
    }

    ImGui::SameLine();

    ImGui::BeginDisabled(!!frame_transfer_);

    if (ImGui::Button("Request Frame")) {
      request_frame();
    }

    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGui::InputText("IPv4", &camera_server_);

    if (frame_transfer_) {
      ImGui::ProgressBar(frame_transfer_->progress());
    }

    if (errored_) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
      ImGui::TextUnformatted("Failed to fetch frame from server.");
      ImGui::PopStyleColor();
    }

    visualizer_->loop();
  }

  void poll() override
  {
    if (!frame_transfer_) {
      return;
    }

    frame_transfer_->poll();

    if (frame_transfer_->done()) {
      if (frame_transfer_->failed()) {
        errored_ = true;
      } else {
        handle_frame();
      }
      frame_transfer_.reset();
    }
  }

protected:
  void handle_frame()
  {
    const auto w{ frame_transfer_->get_header_long("x-image-width",
                                                   /*fallback=*/0L) };

    const auto h{ frame_transfer_->get_header_long("x-image-height",
                                                   /*fallback=*/0L) };

    if ((w <= 0) || (h <= 0) || ((static_cast<size_t>(w) * static_cast<size_t>(h) * 2) != frame_transfer_->size())) {
      return;
    }

    visualizer_->update(frame_transfer_->data(), w, h);
  }

  void request_frame()
  {
    if (frame_transfer_) {
      return;
    }

    std::string url{ std::string("http://") + camera_server_ + ":6500/snapshot" };

    frame_transfer_ = transfer::get(url);

    errored_ = false;
  }

  void save_frame()
  {
    //
  }
};

} // namespace

auto
camera_widget::create() -> std::unique_ptr<camera_widget>
{
  return std::make_unique<camera_widget_impl>();
}

} // namespace cortex
