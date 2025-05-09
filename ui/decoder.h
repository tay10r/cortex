#pragma once

#include <memory>

#include <stdint.h>

namespace cortex {

class buffer;

class frame
{
public:
  virtual ~frame() = default;

  /**
   * @brief Gets the frame data.
   *
   * @details The frame data is 10-bit bayer data. The data is packed so that the
   *          first two channels, which would normally be red and green, is the 2
   *          byte values for the first bayer channel. The 3rd and 4th channel,
   *          which would normally be the blue and alpha channel, is the two byte
   *          values for the bayer channel in the next column.
   * */
  [[nodiscard]] virtual auto data() const -> const uint8_t* = 0;

  [[nodiscard]] virtual auto width() const -> size_t = 0;

  [[nodiscard]] virtual auto height() const -> size_t = 0;
};

class decoder
{
public:
  using Callback = void (*)(void*, std::unique_ptr<frame>);

  static auto create() -> std::unique_ptr<decoder>;

  virtual ~decoder() = default;

  virtual void setup(void* callback_data, Callback callback) = 0;

  virtual void terminate() = 0;

  virtual void loop() = 0;

  virtual void queue(std::unique_ptr<buffer> buffer) = 0;

  [[nodiscard]] virtual auto busy() -> bool = 0;
};

} // namespace cortex
