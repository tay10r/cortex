#pragma once

#include <memory>

namespace cortex {

class tool;

class viewport
{
public:
  static auto create() -> std::unique_ptr<viewport>;

  virtual ~viewport() = default;

  virtual void setup() = 0;

  virtual void teardown() = 0;

  virtual void loop(tool* current_tool) = 0;

  virtual void update_color(const void* rgb, int w, int h) = 0;
};

} // namespace cortex
