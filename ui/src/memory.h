#pragma once

#include <memory>

#include <stddef.h>

namespace cortex {

class memory
{
public:
  static auto create() -> std::unique_ptr<memory>;

  virtual ~memory() = default;

  [[nodiscard]] virtual void* alloc(size_t s) = 0;

  virtual void release(void* addr, size_t s) = 0;

  [[nodiscard]] virtual auto used() const -> size_t = 0;

  [[nodiscard]] virtual auto remaining() const -> size_t = 0;

  [[nodiscard]] virtual auto total() const -> size_t = 0;
};

} // namespace cortex
