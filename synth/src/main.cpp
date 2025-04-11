// For Microsoft compiler to define pi constant:
#define _USE_MATH_DEFINES 1

#include <stb_image_write.h>

#include <FastNoiseLite.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <math.h>
#include <stdint.h>

namespace {

constexpr auto micrometers_per_pixel{ 0.05F };
constexpr auto w{ 4096 };
constexpr auto h{ 2048 };

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
render(const std::vector<cell>& cells, const int w, const int h, uint8_t* rgb, uint8_t* mask)
{
  const auto num_pixels{ w * h };

#pragma omp parallel for

  for (int i = 0; i < num_pixels; i++) {

    const auto xi{ i % w };
    const auto yi{ i / w };

    const auto x{ static_cast<float>(xi) * micrometers_per_pixel };
    const auto y{ static_cast<float>(yi) * micrometers_per_pixel };

    auto pixel{ 0.0F };
    auto m_pixel{ 0 };
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
        if (c.kind != 0) {
          m_pixel = 255;
        }
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

    mask[i] = m_pixel;
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
generate(std::mt19937& rng, const std::filesystem::path& outdir, const int sample_min, const int sample_max)
{
  const char* class_names[]{ "Basophil", "Eosinophil", "Monocyte", "Lymphocyte", "Neutrophil" };

  const auto num_classes{ sizeof(class_names) / sizeof(class_names[0]) };

  std::filesystem::create_directory(outdir);
  std::filesystem::create_directory(outdir / "color");
  std::filesystem::create_directory(outdir / "mask");
  for (size_t i = 0; i < num_classes; i++) {
    std::filesystem::create_directory(outdir / "color" / class_names[i]);
    std::filesystem::create_directory(outdir / "mask" / class_names[i]);
  }

  // simulate bias
  std::uniform_int_distribution<int> count_dist(sample_min, sample_max);
  const std::vector<int> counts{ count_dist(rng), count_dist(rng), count_dist(rng), count_dist(rng), count_dist(rng) };

  const auto x_max{ w * micrometers_per_pixel };
  const auto y_max{ h * micrometers_per_pixel };

  std::uniform_real_distribution<float> r_dist(5, 10);
  std::uniform_real_distribution<float> r2_dist(2, 4);
  std::uniform_real_distribution<float> x_dist(0, x_max);
  std::uniform_real_distribution<float> y_dist(0, y_max);
  std::uniform_int_distribution<int> k_dist(0, 6);
  std::uniform_real_distribution<float> angle_dist(0, 6.28F);
  std::uniform_int_distribution<int> neighbors_dist(0, 8);

  for (auto class_id = 0; class_id < counts.size(); class_id++) {

    const auto num_samples{ counts[class_id] };

    for (auto i = 0; i < num_samples; i++) {

      std::cout << class_names[class_id] << ": " << i << "/" << num_samples << std::endl;

      // note the +1 on the class ID. This is because class 0 is actually a null class (simulates an RBC)
      std::vector<cell> cells{ cell{ w * 0.5F * micrometers_per_pixel,
                                     h * 0.5F * micrometers_per_pixel,
                                     r_dist(rng),
                                     angle_dist(rng),
                                     /*kind=*/(class_id + 1) } };

      const auto num_neighbors{ neighbors_dist(rng) };

      for (auto j = 0; j < num_neighbors; j++) {

        while (true) {

          const auto k = k_dist(rng);
          const auto r = (k == 0) ? r2_dist(rng) : r_dist(rng);

          cell c{ x_dist(rng), y_dist(rng), r, angle_dist(rng), /*kind=*/k };

          if (does_not_overlap(cells, c)) {
            cells.emplace_back(c);
            break;
          }
        }
      }

      std::vector<uint8_t> rgb(w * h * 3, 0);

      std::vector<uint8_t> mask(w * h, 0);

      render(cells, w, h, rgb.data(), mask.data());

      {
        std::ostringstream name_stream;
        name_stream << std::setw(4) << std::setfill('0') << i << ".jpg";
        const auto path = (outdir / "color" / class_names[class_id] / name_stream.str()).string();
        stbi_write_jpg(path.c_str(), w, h, 3, rgb.data(), /*quality=*/70);
      }

      {
        std::ostringstream name_stream;
        name_stream << std::setw(4) << std::setfill('0') << i << ".png";
        const auto path = (outdir / "mask" / class_names[class_id] / name_stream.str()).string();
        stbi_write_png(path.c_str(), w, h, 1, mask.data(), w);
      }
    }
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

  generate(rng, "train", 200, 300);
  generate(rng, "test", 100, 100);

  return EXIT_SUCCESS;
}
