#pragma once

#include "widget.h"

#include <memory>

namespace cortex {

class sangaboard_widget : public widget
{
public:
  static auto create() -> std::unique_ptr<sangaboard_widget>;

  ~sangaboard_widget() override = default;
};

} // namespace cortex
