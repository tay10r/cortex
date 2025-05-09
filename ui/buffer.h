#pragma once

#include <stddef.h>
#include <stdint.h>

namespace cortex {

class buffer
{
public:
  virtual ~buffer() = default;

  [[nodiscard]] virtual auto data() const -> const uint8_t* = 0;

  [[nodiscard]] virtual auto size() const -> size_t = 0;
};

} // namespace cortex
