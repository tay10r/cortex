#include <imgui.h>

#include <uikit/main.hpp>

#include "src/camera_widget.h"
#include "src/gallery_widget.h"
#include "src/image_index.h"
#include "src/sangaboard_widget.h"
#include "src/tools_widget.h"

#include <map>
#include <string>

namespace {

using namespace cortex;

class app final : public uikit::app
{
  size_t next_id_{ 1 };

  std::shared_ptr<image_index> image_index_{ image_index::create() };

  std::map<std::string, std::unique_ptr<camera_widget>> camera_widgets_;

  std::map<std::string, std::unique_ptr<sangaboard_widget>> sangaboard_widgets_;

  std::map<std::string, std::unique_ptr<gallery_widget>> gallery_widgets_;

  std::unique_ptr<tools_widget> tools_widget_;

public:
  void setup(uikit::platform& plt) override
  {
    plt.set_app_name("Cortex");

    image_index_->refresh();
  }

  void teardown(uikit::platform&) override
  {
    for (auto& cam : camera_widgets_) {
      cam.second->teardown();
    }

    for (auto& sg : sangaboard_widgets_) {
      sg.second->teardown();
    }
  }

  void loop(uikit::platform&) override
  {
    image_index_->loop();

    ImGui::DockSpaceOverViewport();

    if (ImGui::BeginMainMenuBar()) {

      render_main_menu();

      ImGui::EndMainMenuBar();
    }

    update_widgets(sangaboard_widgets_);

    update_widgets(camera_widgets_);

    update_widgets(gallery_widgets_);

    update_widget("Tools", tools_widget_, ImVec2(256, 512));
  }

protected:
  template<typename T>
  void update_widget(const char* label, std::unique_ptr<T>& widget_ptr, const ImVec2& initial_size)
  {
    if (!widget_ptr) {
      return;
    }

    auto* w = static_cast<widget*>(widget_ptr.get());

    w->poll();

    bool open{ true };

    ImGui::SetNextWindowSize(initial_size, ImGuiCond_Appearing);

    if (ImGui::Begin(label, &open)) {
      w->render();
    }
    ImGui::End();

    if (!open) {
      w->teardown();
      widget_ptr.reset();
    }
  }

  template<typename T>
  static void update_widgets(std::map<std::string, std::unique_ptr<T>>& widgets)
  {
    for (auto& entry : widgets) {
      // poll IO
      // called each frame, regardless of visibility
      entry.second->poll();
    }

    std::vector<std::string> to_delete;

    for (auto& entry : widgets) {

      bool open{ true };

      ImGui::SetNextWindowSize(ImVec2(512, 512), ImGuiCond_Appearing);

      if (ImGui::Begin(entry.first.c_str(), &open)) {
        entry.second->render();
      }

      if (!open) {
        to_delete.emplace_back(entry.first);
      }

      ImGui::End();
    }

    for (size_t j = to_delete.size(); j > 0; --j) {
      widgets.at(to_delete.at(j - 1))->teardown();
      widgets.erase(to_delete.at(j - 1));
    }
  }

  void render_main_menu()
  {
    if (ImGui::BeginMenu("Widgets")) {

      if (ImGui::MenuItem("New Camera Widget")) {
        auto widget = camera_widget::create(this, on_plot);
        widget->setup();
        camera_widgets_.emplace("Camera ##" + std::to_string(generate_id()), std::move(widget));
      }

      if (ImGui::MenuItem("New Sangaboard Widget")) {
        auto widget = sangaboard_widget::create();
        widget->setup();
        sangaboard_widgets_.emplace("Sangaboard ##" + std::to_string(generate_id()), std::move(widget));
      }

      if (ImGui::MenuItem("New Gallery Widget")) {
        auto widget = gallery_widget::create(this, on_plot, image_index_);
        widget->setup();
        gallery_widgets_.emplace("Gallery ##" + std::to_string(generate_id()), std::move(widget));
      }

      if (ImGui::MenuItem("New Tools Widget")) {
        auto widget = tools_widget::create();
        widget->setup();
        tools_widget_ = std::move(widget);
      }

      ImGui::EndMenu();
    }
  }

  [[nodiscard]] auto generate_id() -> size_t { return next_id_++; }

  static void on_plot(void* self_ptr, const image& img)
  {
    auto* self = static_cast<app*>(self_ptr);

    if (self->tools_widget_) {
      self->tools_widget_->plot(img);
    }
  }
};

} // namespace

namespace uikit {

auto
app::create() -> std::unique_ptr<app>
{
  return std::make_unique<::app>();
}

} // namespace uikit
