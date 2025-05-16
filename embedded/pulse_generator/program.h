#include <stddef.h>

[[nodiscard]]
constexpr auto
max_line_size() -> size_t
{
  return 255;
};

class program final
{
  char line_buffer_[max_line_size() + 1]{};

  size_t line_size_{};

public:
  [[nodiscard]] auto put(char c) -> bool;

  void reset();

  [[nodiscard]] auto parse_pulse_duration(int* duration) const -> bool;

  [[nodiscard]] auto parse_on() const -> bool;

  [[nodiscard]] auto parse_off() const -> bool;

  [[nodiscard]] auto parse_help() const -> bool;
};
