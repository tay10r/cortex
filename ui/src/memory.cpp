#include "memory.h"

#include <stdlib.h>

namespace cortex {

namespace {

class memory_impl final : public memory
{
  size_t total_{};

  size_t used_{};

public:
  void* alloc(const size_t s) override
  {
    if ((used_ + s) > total_) {
      return nullptr;
    }
    used_ += s;
    return malloc(s);
  }

  void release(void* addr, const size_t s) override
  {
    used_ -= s;
    free(addr);
  }

  auto remaining() const -> size_t override { return (used_ < total_) ? total_ - used_ : 0; }

  auto total() const -> size_t override { return total_; }
};

} // namespace

auto
memory::create() -> std::unique_ptr<memory>
{
  return std::make_unique<memory_impl>();
}

} // namespace cortex
