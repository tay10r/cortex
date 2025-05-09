#pragma once

#include "widget.h"

#include "image_index.h"

#include <memory>

namespace cortex {

class gallery_widget : public widget
{
public:
  static auto create(std::shared_ptr<image_index> img_index) -> std::unique_ptr<gallery_widget>;

  ~gallery_widget() override = default;
};

} // namespace cortex
