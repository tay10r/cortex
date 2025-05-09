#pragma once

#include <memory>
#include <string>
#include <vector>

#include <stdint.h>

namespace cortex {

struct image_info final
{
  std::string id;

  size_t width{};

  size_t height{};

  uint64_t creation_time{};

  std::string label;
};

class image_index
{
public:
  static auto create() -> std::unique_ptr<image_index>;

  virtual ~image_index() = default;

  virtual void loop() = 0;

  [[nodiscard]] virtual auto refreshing() const -> bool = 0;

  virtual void refresh() = 0;

  [[nodiscard]] virtual auto get_image_info() const -> const std::vector<image_info>& = 0;

  [[nodiscard]] virtual auto get_image_info() -> std::vector<image_info>& = 0;
};

} // namespace cortex
