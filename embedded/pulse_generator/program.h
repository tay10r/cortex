#include <stddef.h>

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
  auto put(char c) -> bool;

  void reset();

  auto parse_pulse(unsigned int* delay, unsigned int* duration) const -> bool;

  auto parse_pwm(int* duty_cycle) const -> bool;

  auto parse_on() const -> bool;

  auto parse_off() const -> bool;

  auto parse_help() const -> bool;
};
