#include "gallery_widget.h"

#include "task.h"
#include "visualizer.h"

#include <imgui.h>

#include <sstream>

namespace cortex {

namespace {

class gallery_widget_impl final : public gallery_widget
{
  std::shared_ptr<image_index> image_index_;

  size_t selected_{ static_cast<size_t>(-1) };

  std::unique_ptr<visualizer> visualizer_;

  std::unique_ptr<task> image_task_;

  std::unique_ptr<task> delete_task_;

  std::string error_;

public:
  explicit gallery_widget_impl(void* parent, plot_callback plot_cb, std::shared_ptr<image_index> img_index)
    : image_index_(std::move(img_index))
    , visualizer_(visualizer::create(parent, plot_cb))
  {
  }

  void setup() override { visualizer_->setup(); }

  void teardown() override { visualizer_->teardown(); }

  void render() override
  {
    ImGui::BeginDisabled(image_index_->refreshing());

    if (ImGui::Button("Refresh")) {
      image_index_->refresh();
    }

    ImGui::EndDisabled();

    ImGui::SameLine();

    render_image_selection_list();

    ImGui::SameLine();

    const auto& imgs = image_index_->get_image_info();
    const auto has_selection{ selected_ < imgs.size() };
    ImGui::BeginDisabled(!has_selection || !!delete_task_);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1, 0, 0, 1));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
    if (ImGui::Button("Delete")) {
      delete_image();
    }
    ImGui::PopStyleColor(2);
    ImGui::EndDisabled();

    if (!error_.empty()) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0, 0, 1));
      ImGui::TextUnformatted(error_.c_str());
      ImGui::PopStyleColor();
    }

    visualizer_->loop();
  }

  void poll() override
  {
    if (image_task_) {

      image_task_->poll();

      if (image_task_->done()) {

        if (image_task_->failed()) {
          error_ = "Failed to fetch image data.";
        } else {
          handle_image(image_task_->data(), image_task_->size());
        }

        image_task_.reset();
      }
    }

    if (delete_task_) {
      delete_task_->poll();
      if (delete_task_->done()) {
        if (!delete_task_->failed()) {
          image_index_->refresh();
          selected_ = static_cast<size_t>(-1);
        }
        delete_task_.reset();
      }
    }
  }

protected:
  [[nodiscard]] auto has_selection() const -> bool { return selected_ < image_index_->get_image_info().size(); }

  void handle_image(const void* data, const size_t size)
  {
    const auto& info = image_index_->get_image_info().at(selected_);

    const auto expected_size{ info.width * info.height * 2 };

    if (expected_size != size) {
      error_ = "Image size does not match dimensions in index.";
      return;
    }

    visualizer_->update(data, info.width, info.height);
  }

  void get_image()
  {
    error_.clear();

    if (image_task_ || !has_selection()) {
      return;
    }

    image_task_ = task::http_get("/images/" + image_index_->get_image_info().at(selected_).id + ".bin");
  }

  void delete_image()
  {
    if (!has_selection() || !!delete_task_) {
      return;
    }

    const auto& imgs = image_index_->get_image_info();

    std::ostringstream url_stream;
    url_stream << "/images/" << imgs.at(selected_).id << ".bin";
    delete_task_ = task::http_delete(url_stream.str());
  }

  void render_image_selection_list()
  {
    const auto& imgs = image_index_->get_image_info();

    const auto has_selection{ selected_ < imgs.size() };

    const auto* preview{ has_selection ? imgs.at(selected_).label.c_str() : "" };

    if (ImGui::BeginCombo("##Selected", preview)) {

      ImGui::BeginDisabled(!!image_task_);

      for (size_t i = 0; i < imgs.size(); i++) {

        const auto& img = imgs[i];

        const auto selected{ i == selected_ };

        if (ImGui::Selectable(img.label.c_str(), selected)) {
          selected_ = i;
          get_image();
        }
      }

      ImGui::EndDisabled();

      ImGui::EndCombo();
    }
  }
};

} // namespace

auto
gallery_widget::create(void* parent, plot_callback plot_cb, std::shared_ptr<image_index> img_index)
  -> std::unique_ptr<gallery_widget>
{
  return std::make_unique<gallery_widget_impl>(parent, plot_cb, std::move(img_index));
}

} // namespace cortex
