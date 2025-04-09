#include <stb_image_write.h>

#include <FastNoiseLite.h>

#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#include <math.h>
#include <stdint.h>

namespace {

const auto micrometers_per_pixel{ 0.2F };

using rng_type = std::mt19937;

struct cell final
{
  float x;
  float y;
  float r;
  float angle;
  int kind;
};

FastNoiseLite noise0{ 1 };
FastNoiseLite noise1{ 2 };
FastNoiseLite noise2{ 3 };
FastNoiseLite noise3{ 4 };
FastNoiseLite noise4{ 5 };
FastNoiseLite noise5{ 6 };
FastNoiseLite noise6{ 7 };

auto
shade(const cell& c, const float theta, const float r) -> float
{
  const auto x{ r * cosf(theta) };
  const auto y{ r * sinf(theta) };
  const auto z{ sqrtf(1 - r * r) };

  float result{ -1.0F };

  switch (c.kind) {
    case 0:
      result = noise0.GetNoise(x, y, z);
      break;
    case 1:
      result = noise1.GetNoise(x, y, z);
      break;
    case 2:
      result = noise2.GetNoise(x, y, z);
      break;
    case 3:
      result = noise3.GetNoise(x, y, z);
      break;
    case 4:
      result = noise4.GetNoise(x, y, z);
      break;
    case 5:
      result = noise5.GetNoise(x, y, z);
      break;
    case 6:
      result = noise6.GetNoise(x, y, z);
      break;
  }

  return result * 0.5F + 0.5F;
}

void
render(const std::vector<cell>& cells, const int w, const int h, uint8_t* rgb)
{
  const auto num_pixels{ w * h };

#pragma omp parallel for

  for (int i = 0; i < num_pixels; i++) {

    const auto xi{ i % w };
    const auto yi{ i / w };

    const auto x{ static_cast<float>(xi) * micrometers_per_pixel };
    const auto y{ static_cast<float>(yi) * micrometers_per_pixel };

    auto pixel{ 0 };

    auto k{ 0 };

    for (const auto& c : cells) {

      const auto dx{ c.x - x };
      const auto dy{ c.y - y };

      const auto r{ dx * dx + dy * dy };

      if (r < (c.r * c.r)) {
        const auto nr{ sqrtf(r) / c.r };
        const auto mag{ sqrtf(dx * dx + dy * dy) };
        const auto ndx{ dx / mag };
        const auto ndy{ dy / mag };
        const auto angle = fmodf(c.angle + atan2(ndy, ndx), 2.0F * M_PI);
        pixel += shade(c, angle, nr);
        k += 1;
      }
    }

    if (k > 0) {
      pixel /= static_cast<float>(k);
    }

    const auto p = static_cast<int>(pixel * 255);

    rgb[i * 3 + 0] = p;
    rgb[i * 3 + 1] = p;
    rgb[i * 3 + 2] = p;
  }
}

[[nodiscard]] auto
does_not_overlap(const std::vector<cell>& cells, const cell& c) -> bool
{
  for (const auto& other_c : cells) {
    const auto dx{ other_c.x - c.x };
    const auto dy{ other_c.y - c.y };
    const auto r{ sqrtf(dx * dx + dy * dy) };
    if (r < (c.r + other_c.r)) {
      return false;
    }
  }

  return true;
}

void
generate(const int num_frames, std::mt19937& rng, const std::filesystem::path& outdir)
{
  std::filesystem::create_directory(outdir);

  for (auto i = 0; i < num_frames; i++) {
  }
}

} // namespace

auto
main() -> int
{
  const auto seed{ 0 };

  const auto freq{ 0.5F };
  noise0.SetFrequency(freq);
  noise1.SetFrequency(freq);
  noise2.SetFrequency(freq);
  noise3.SetFrequency(freq);
  noise4.SetFrequency(freq);
  noise5.SetFrequency(freq);
  noise6.SetFrequency(freq);

  const auto octaves{ 2 };
  noise0.SetFractalOctaves(octaves);
  noise1.SetFractalOctaves(octaves);
  noise2.SetFractalOctaves(octaves);
  noise3.SetFractalOctaves(octaves);
  noise4.SetFractalOctaves(octaves);
  noise5.SetFractalOctaves(octaves);
  noise6.SetFractalOctaves(octaves);

  const auto fractal{ FastNoiseLite::FractalType_Ridged };
  noise0.SetFractalType(fractal);
  noise1.SetFractalType(fractal);
  noise2.SetFractalType(fractal);
  noise3.SetFractalType(fractal);
  noise4.SetFractalType(fractal);
  noise5.SetFractalType(fractal);
  noise6.SetFractalType(fractal);

  rng_type rng(seed);

  std::uniform_real_distribution<float> r_dist(5, 20);
  // std::uniform_real_distribution<float> x_dist(20, 80);
  // std::uniform_real_distribution<float> y_dist(20, 80);
  std::uniform_real_distribution<float> x_dist(5, 40);
  std::uniform_real_distribution<float> y_dist(5, 40);
  std::uniform_int_distribution<int> k_dist(0, 6);
  std::uniform_real_distribution<float> angle_dist(0, 6.28F);

  const auto w{ 575 };
  const auto h{ 575 };

  const auto num_train_frames{ 16 };

  const auto num_test_frames{ 16 };

  for (auto i = 0; i < num_test_frames; i++) {
  }

  std::vector<cell> cells;

  const auto num_cells{ 5 };

  for (auto i = 0; i < num_cells; i++) {

    while (true) {

      cell c{ x_dist(rng), y_dist(rng), r_dist(rng), angle_dist(rng), /*kind=*/k_dist(rng) };

      if (does_not_overlap(cells, c)) {
        std::cout << c.x << ' ' << c.y << ' ' << c.r << std::endl;
        cells.emplace_back(c);
        break;
      }
    }
  }

  std::vector<uint8_t> rgb(w * h * 3, 0);

  render(cells, w, h, rgb.data());

  stbi_write_png("result.png", w, h, 3, rgb.data(), w * 3);

  return EXIT_SUCCESS;
}
