#pragma once

#include "plot.h"
#include "widget.h"

#include <memory>

namespace cortex {

class camera_widget : public widget
{
public:
  static auto create(void* parent, plot_callback plot_cb) -> std::unique_ptr<camera_widget>;

  ~camera_widget() override = default;
};

} // namespace cortex
