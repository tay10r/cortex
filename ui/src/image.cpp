#include "image.h"

#include <stdlib.h>
#include <string.h>

namespace cortex {

namespace {

[[nodiscard]] auto
shift_bayer_format(const pixel_format base, const size_t dx, const size_t dy) -> pixel_format
{
  using pf = pixel_format;

  const auto shift = (dy % 2) * 2 + (dx % 2); // maps to [0, 3]
                                              //
  static const pixel_format lut[4][4] = {
    // clang-format off
    // rggb
    { pixel_format::rggb, pixel_format::grbg, pixel_format::gbrg, pixel_format::bggr },
    // grbg
    { pixel_format::grbg, pixel_format::rggb, pixel_format::bggr, pixel_format::gbrg },
    // gbrg
    { pixel_format::gbrg, pixel_format::bggr, pixel_format::rggb, pixel_format::grbg },
    // bggr
    { pixel_format::bggr, pixel_format::gbrg, pixel_format::grbg, pixel_format::rggb }
    // clang-format on
  };

  return lut[static_cast<int>(base)][shift];
}

} // namespace

image_view::image_view(const uint16_t* data, const size_t w, const size_t h, const size_t p, const pixel_format pf)
  : data_(data)
  , width_(w)
  , height_(h)
  , pitch_(p)
  , format_(pf)
{
}

image::image(const uint16_t* data, const size_t w, const size_t h, const pixel_format pf)
{
  data_ = static_cast<uint16_t*>(malloc(w * h * 2));
  if (data_) {
    memcpy(data_, data, w * h * 2);
    width_ = w;
    height_ = h;
    format_ = pf;
  }
}

image::~image()
{
  free(data_);
}

image::image(image&& other)
  : data_(other.data_)
  , width_(other.width_)
  , height_(other.height_)
  , format_(other.format_)
{
  other.data_ = nullptr;
  other.width_ = 0;
  other.height_ = 0;
}

auto
image::operator=(image&& other) -> image&
{
  free(data_);

  data_ = other.data_;
  width_ = other.width_;
  height_ = other.height_;
  format_ = other.format_;

  other.data_ = nullptr;
  other.width_ = 0;
  other.height_ = 0;

  return *this;
}

auto
image::view(const size_t x, const size_t y, const size_t w, const size_t h) const -> image_view
{
  return image_view(data_ + y * width_ + x, w, h, width_, shift_bayer_format(format_, x, y));
}

} // namespace cortex
