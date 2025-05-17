#include "program.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

auto
program::put(const char c) -> bool
{
  if (c == '\n') {
    return true;
  }

  if (line_size_ == max_line_size()) {
    reset();
  }

  line_buffer_[line_size_] = c;
  line_buffer_[line_size_ + 1] = 0;

  line_size_++;

  return false;
}

void
program::reset()
{
  line_buffer_[0] = 0;
  line_size_ = 0;
}

auto
program::parse_pwm(int* duty_cycle) const -> bool
{
  constexpr size_t k{ sizeof("pwm") - 1 };

  if (line_size_ < k) {
    return false;
  }

  if (memcmp(line_buffer_, "pwm", k) != 0) {
    return false;
  }

  const char* args{ line_buffer_ + k };

  if (sscanf(args, "%d", duty_cycle) != 1) {
    return false;
  }
  return (*duty_cycle >= 0) && (*duty_cycle <= 255);
}

auto
program::parse_pulse(unsigned int* delay, unsigned int* duration) const -> bool
{
  constexpr size_t k{ sizeof("pulse") - 1 };

  if (line_size_ < k) {
    return false;
  }

  if (memcmp(line_buffer_, "pulse", k) != 0) {
    return false;
  }

  const char* args{ line_buffer_ + k };

  if (memchr(line_buffer_, '-', line_size_) != nullptr) {
    return false;
  }

  return sscanf(args, "%u %u", delay, duration) == 2;
}

auto
program::parse_on() const -> bool
{
  return strcmp(line_buffer_, "on") == 0;
}

auto
program::parse_off() const -> bool
{
  return strcmp(line_buffer_, "off") == 0;
}

auto
program::parse_help() const -> bool
{
  return strcmp(line_buffer_, "help") == 0;
}
