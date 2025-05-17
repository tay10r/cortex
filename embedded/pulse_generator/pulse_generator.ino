#include "program.h"

constexpr int led_pin { 12 };

program g_program;

void
setup()
{
  SerialUSB.begin(115200);
  pinMode(led_pin, OUTPUT);
  digitalWrite(led_pin, LOW);
}

namespace {

constexpr char help[] = R"(
This program is primarily for illuminating a slide for a period period of time.
There are a couple of commands:

                          "help" : Prints this help message.
                            "on" : Turns the LED on.
                           "off" : Turns the LED off.
      "pulse" <delay> <duration> : Delays for a certain number of microseconds,
                                   then activates the LED for a certain number
                                   of microseconds.
              "pwm" <duty-cycle> : Activates a PWM signal on the LED pin.
)";

void handle_command()
{
  static_assert(sizeof(int) == 4, "size of int must be 4 bytes.");
  unsigned int delay {};
  unsigned int duration {};
  int duty_cycle {};
  if (g_program.parse_pulse(&delay, &duration)) {
    pinMode(led_pin, OUTPUT);
    // TODO : wait for trigger from FSTROBE (with timeout)
    const auto t0 = micros();
    delayMicroseconds(delay);
    const auto t1 = micros();
    digitalWrite(led_pin, HIGH);
    const auto t2 = micros();
    delayMicroseconds(duration);
    const auto t3 = micros();
    digitalWrite(led_pin, LOW);
    const auto t4 = micros();
    SerialUSB.println(t0);
    SerialUSB.println(t1);
    SerialUSB.println(t2);
    SerialUSB.println(t3);
    SerialUSB.println(t4);
    SerialUSB.println();
  } else if (g_program.parse_pwm(&duty_cycle)) {
    pinMode(led_pin, OUTPUT);
    analogWrite(led_pin, duty_cycle);
  } else if (g_program.parse_on()) {
    pinMode(led_pin, OUTPUT);
    digitalWrite(led_pin, HIGH);
    SerialUSB.println();
  } else if (g_program.parse_off()) {
    pinMode(led_pin, OUTPUT);
    digitalWrite(led_pin, LOW);
    SerialUSB.println();
  } else if (g_program.parse_help()) {
    SerialUSB.print(help);
    SerialUSB.println();
  } else {
    SerialUSB.println("bad command.");
    SerialUSB.println();
  }
  g_program.reset();
}

} // namespace

void
loop()
{
  while (SerialUSB.available() > 0) {
    const auto c = SerialUSB.read();
    if (c < 0) {
      break;
    }
    if (g_program.put(static_cast<char>(c))) {
      handle_command();
    }
  }
}
