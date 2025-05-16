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
  "<number>" : Activates the LED for a certain number of microseconds.
)";

void handle_command()
{
  static_assert(sizeof(int) == 4, "size of int must be 4 bytes.");
  int duration {};
  if (g_program.parse_pulse_duration(&duration)) {
    if (duration >= 0) {
      const auto t0 = micros();
      digitalWrite(led_pin, HIGH);
      const auto t1 = micros();
      delayMicroseconds(duration);
      const auto t2 = micros();
      digitalWrite(led_pin, LOW);
      const auto t3 = micros();
      SerialUSB.println(t0);
      SerialUSB.println(t1);
      SerialUSB.println(t2);
      SerialUSB.println(t3);
      SerialUSB.println();
    } else {
      SerialUSB.println("delay cannot be negative.");
      SerialUSB.println();
    }
  } else if (g_program.parse_on()) {
    digitalWrite(led_pin, HIGH);
    SerialUSB.println();
  } else if (g_program.parse_off()) {
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
