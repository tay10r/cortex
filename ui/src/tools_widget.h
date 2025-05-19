#pragma once

#include "widget.h"

#include <memory>

namespace cortex {

class image;

class tools_widget : public widget
{
public:
  static auto create() -> std::unique_ptr<tools_widget>;

  virtual ~tools_widget() = default;

  virtual void plot(const image& img) = 0;
};

} // namespace cortex
