#pragma once

namespace cortex {

class widget
{
public:
  virtual ~widget() = default;

  virtual void setup() = 0;

  virtual void teardown() = 0;

  /**
   * @brief If the parent window is visible, will render the widget.
   * */
  virtual void render() = 0;

  /**
   * @brief Checks for IO updates, must be called every frame.
   * */
  virtual void poll() = 0;
};

} // namespace cortex
