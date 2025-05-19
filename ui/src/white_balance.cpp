#include "white_balance.h"

#include "image.h"

namespace cortex {

namespace {

void
bin_rg(const uint16_t* pixels, const size_t count, histogram& hist)
{
  for (size_t i = 0; i < (count / 2); i++) {
    const auto r = pixels[i * 2 + 0] >> 2;
    const auto g = pixels[i * 2 + 1] >> 2;
    hist.r[r] += 1.0F;
    hist.g[g] += 1.0F;
  }
}

void
bin_gr(const uint16_t* pixels, const size_t count, histogram& hist)
{
  for (size_t i = 0; i < (count / 2); i++) {
    const auto g = pixels[i * 2 + 0] >> 2;
    const auto r = pixels[i * 2 + 1] >> 2;
    hist.g[g] += 1.0F;
    hist.r[r] += 1.0F;
  }
}

void
bin_bg(const uint16_t* pixels, const size_t count, histogram& hist)
{
  for (size_t i = 0; i < (count / 2); i++) {
    const auto b = pixels[i * 2 + 0] >> 2;
    const auto g = pixels[i * 2 + 1] >> 2;
    hist.b[b] += 1.0F;
    hist.g[g] += 1.0F;
  }
}

void
bin_gb(const uint16_t* pixels, const size_t count, histogram& hist)
{
  for (size_t i = 0; i < (count / 2); i++) {
    const auto g = pixels[i * 2 + 0] >> 2;
    const auto b = pixels[i * 2 + 1] >> 2;
    hist.g[g] += 1.0F;
    hist.b[b] += 1.0F;
  }
}

} // namespace

void
histogram::reset()
{
  for (size_t i = 0; i < 256; i++) {
    r[i] = 0.0F;
    g[i] = 0.0F;
    b[i] = 0.0F;
  }
}

namespace {

auto
value_at_quantile(const float* hist, const float quantile) -> float
{
  auto total{ 0.0f };

  for (size_t i = 0; i < 256; ++i) {
    total += hist[i];
  }

  const auto threshold = total * quantile;

  auto cumulative{ 0.0f };

  for (size_t i = 0; i < 256; ++i) {

    cumulative += hist[i];

    if (cumulative >= threshold) {
      return static_cast<float>(i) / 255.0F;
    }
  }

  return 1.0f; // fallback if empty
}

} // namespace

void
histogram::compute_quantile_values(const float quantile, float* rgb) const
{
  rgb[0] = value_at_quantile(r, quantile);
  rgb[1] = value_at_quantile(g, quantile);
  rgb[2] = value_at_quantile(b, quantile);
}

void
compute_histogram(const image_view& img, histogram& hist)
{
  switch (img.format()) {
    case pixel_format::rggb:
      for (size_t y = 0; y < (img.height() / 2); y++) {
        bin_rg(img.row(y * 2 + 0), img.width(), hist);
        bin_gb(img.row(y * 2 + 1), img.width(), hist);
      }
      break;
    case pixel_format::bggr:
      for (size_t y = 0; y < (img.height() / 2); y++) {
        bin_bg(img.row(y * 2 + 0), img.width(), hist);
        bin_gr(img.row(y * 2 + 1), img.width(), hist);
      }
      break;
    case pixel_format::gbrg:
      for (size_t y = 0; y < (img.height() / 2); y++) {
        bin_gb(img.row(y * 2 + 0), img.width(), hist);
        bin_rg(img.row(y * 2 + 1), img.width(), hist);
      }
      break;
    case pixel_format::grbg:
      for (size_t y = 0; y < (img.height() / 2); y++) {
        bin_gr(img.row(y * 2 + 0), img.width(), hist);
        bin_bg(img.row(y * 2 + 1), img.width(), hist);
      }
      break;
  }
}

} // namespace cortex
