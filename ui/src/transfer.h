#pragma once

#include <memory>
#include <string>

#include <stdlib.h>

namespace cortex {

class transfer
{
public:
  static auto get(const std::string& url) -> std::unique_ptr<transfer>;

  static auto post(const std::string& url, std::string body) -> std::unique_ptr<transfer>;

  static auto delete_(const std::string& url) -> std::unique_ptr<transfer>;

  static auto put(const std::string& url) -> std::unique_ptr<transfer>;

  virtual ~transfer() = default;

  virtual void poll() = 0;

  [[nodiscard]] virtual auto progress() const -> float = 0;

  [[nodiscard]] virtual auto done() const -> bool = 0;

  [[nodiscard]] virtual auto failed() const -> bool = 0;

  [[nodiscard]] virtual auto data() const -> const void* = 0;

  [[nodiscard]] virtual auto size() const -> size_t = 0;

  [[nodiscard]] virtual auto get_header_value(const char* key) const -> const char* = 0;

  [[nodiscard]] auto get_header_long(const char* key, const long fallback) const -> long
  {
    const auto* value{ get_header_value(key) };
    return value ? atol(value) : fallback;
  }
};

} // namespace cortex
