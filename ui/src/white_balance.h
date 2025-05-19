#pragma once

namespace cortex {

class image_view;

struct histogram final
{
  float r[256];

  float g[256];

  float b[256];

  void reset();

  void compute_quantile_values(float quantile, float* rgb) const;
};

void
compute_histogram(const image_view& img, histogram& hist);

} // namespace cortex
