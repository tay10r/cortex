#include <gtest/gtest.h>

#include <random>

#include "../program.h"

namespace {

auto
put_all(program& prg, const char* str) -> bool
{
  while (*str) {
    if (prg.put(*str)) {
      return true;
    }
    str++;
  }
  return false;
}

} // namespace

TEST(Program, ParseHelpCommand)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "help\n"));
  EXPECT_TRUE(prg.parse_help());
}

TEST(Program, ParsePwmCommand)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pwm 32\n"));
  int duty_cycle{};
  EXPECT_TRUE(prg.parse_pwm(&duty_cycle));
  EXPECT_EQ(duty_cycle, 32);
}

TEST(Program, ParseBadPwmCommand)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pwm\n"));
  int duty_cycle{};
  EXPECT_FALSE(prg.parse_pwm(&duty_cycle));
}

TEST(Program, ParseBadPwmCommand2)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pwm -32\n"));
  int duty_cycle{};
  EXPECT_FALSE(prg.parse_pwm(&duty_cycle));
}

TEST(Program, ParseBadPwmCommand3)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pwm 256\n"));
  int duty_cycle{};
  EXPECT_FALSE(prg.parse_pwm(&duty_cycle));
}

TEST(Program, ParsePulseCommand)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pulse 52 1025\n"));
  unsigned int delay{};
  unsigned int duration{};
  EXPECT_TRUE(prg.parse_pulse(&delay, &duration));
  EXPECT_EQ(delay, 52);
  EXPECT_EQ(duration, 1025);
}

TEST(Program, ParseBadPulseCommand)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pulse\n"));
  unsigned int delay{};
  unsigned int duration{};
  EXPECT_FALSE(prg.parse_pulse(&delay, &duration));
}

TEST(Program, ParseBadPulseCommand2)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pulse 100\n"));
  unsigned int delay{};
  unsigned int duration{};
  EXPECT_FALSE(prg.parse_pulse(&delay, &duration));
}

TEST(Program, ParseBadPulseCommand3)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pulse -100 1000\n"));
  unsigned int delay{};
  unsigned int duration{};
  EXPECT_FALSE(prg.parse_pulse(&delay, &duration));
  EXPECT_EQ(delay, 0);
}

TEST(Program, ParseBadPulseCommand4)
{
  program prg;
  ASSERT_TRUE(put_all(prg, "pulse 100 -1000\n"));
  unsigned int delay{};
  unsigned int duration{};
  EXPECT_FALSE(prg.parse_pulse(&delay, &duration));
}

TEST(Program, FuzzTest)
{
  std::mt19937 rng;

  const size_t num_iterations{ 8192 };

  std::uniform_int_distribution<int> dist(-128, 127);

  program prg;

  for (size_t i = 0; i < num_iterations; i++) {
    // We're not looking for anything specifically other than
    // to verify that the program does crash or report any errors
    // when run under valgrind.
    (void)prg.put(static_cast<char>(dist(rng)));
  }
}
