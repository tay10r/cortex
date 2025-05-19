#pragma once

#include <stddef.h>
#include <stdint.h>

namespace cortex {

enum class pixel_format
{
  rggb,
  gbrg,
  grbg,
  bggr
};

class image_view final
{
  const uint16_t* data_{};

  size_t width_{};

  size_t height_{};

  size_t pitch_{};

  pixel_format format_{};

public:
  image_view() = default;

  image_view(const uint16_t* data, size_t w, size_t h, size_t p, pixel_format pf);

  [[nodiscard]] auto row(const size_t y) const -> const uint16_t* { return data_ + pitch_ * y; }

  [[nodiscard]] auto width() const -> size_t { return width_; }

  [[nodiscard]] auto height() const -> size_t { return height_; }

  [[nodiscard]] auto format() const -> pixel_format { return format_; }
};

class image final
{
  uint16_t* data_{};

  size_t width_{};

  size_t height_{};

  pixel_format format_{};

public:
  image() = default;

  image(const uint16_t* data, size_t w, size_t h, pixel_format pf = pixel_format::rggb);

  image(const image&) = delete;

  image(image&&);

  ~image();

  auto operator=(const image&) -> image& = delete;

  auto operator=(image&&) -> image&;

  [[nodiscard]] auto width() const -> size_t { return width_; }

  [[nodiscard]] auto height() const -> size_t { return height_; }

  [[nodiscard]] auto data() -> uint16_t* { return data_; }

  [[nodiscard]] auto data() const -> const uint16_t* { return data_; }

  [[nodiscard]] auto format() const -> pixel_format { return format_; }

  [[nodiscard]] auto view(size_t x, size_t y, size_t w, size_t h) const -> image_view;
};

} // namespace cortex
