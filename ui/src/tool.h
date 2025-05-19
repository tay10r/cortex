#pragma once

namespace cortex {

class image;
class image_view;

class tool
{
public:
  virtual ~tool() = default;

  virtual void render_ui() = 0;

  virtual void render_plot(const image& img, const long x, const long y) = 0;

  virtual void on_select() = 0;

  virtual void on_deselect() = 0;

protected:
  [[nodiscard]] static auto plot_mouse_box(const image& img, const long x, const long y, const long box_size) -> bool;

  [[nodiscard]] static auto is_box_valid(const image& img, const long x, const long y, const long box_size) -> bool;

  [[nodiscard]] static auto crop_box(const image& img, const long x, const long y, const long box_size) -> image_view;
};

} // namespace cortex
