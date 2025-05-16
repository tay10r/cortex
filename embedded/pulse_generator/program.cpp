#include "program.h"

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
program::parse_pulse_duration(int* duration) const -> bool
{
  if (line_size_ == 0) {
    return false;
  }

  const auto first = line_buffer_[0];
  if ((first >= '0') && (first <= '9')) {
    *duration = atoi(line_buffer_);
    return true;
  } else {
    return false;
  }
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