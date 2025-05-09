#pragma once

#include "widget.h"

#include <memory>

namespace cortex {

class camera_widget : public widget
{
public:
  static auto create() -> std::unique_ptr<camera_widget>;

  ~camera_widget() override = default;
};

} // namespace cortex
