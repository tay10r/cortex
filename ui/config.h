#pragma once

#include <memory>

namespace cortex {

class config_widget
{
public:
  static auto create() -> std::unique_ptr<config_widget>;

  virtual ~config_widget() = default;

  virtual void loop() = 0;
};

} // namespace cortex
