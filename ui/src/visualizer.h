#pragma once

#include <memory>

namespace cortex {

/**
 * @brief This class is responsible for taking the bayer data and converting it
 * to a visual representation.
 * */
class visualizer
{
public:
  static auto create() -> std::unique_ptr<visualizer>;

  virtual ~visualizer() = default;

  virtual void setup() = 0;

  virtual void teardown() = 0;

  virtual void loop() = 0;

  virtual void update(const void* bayer_data, int w, int h) = 0;
};

} // namespace cortex
