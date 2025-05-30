import serial


class PulseController:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.__device = serial.Serial(port, baudrate, timeout=1)

    def read_response(self) -> list[str]:
        result: list[str] = []
        while True:
            line = self.__device.readline()
            if line == b'':
                break
            data = line.decode().strip()
            if data == '':
                break
            result.append(data)
        return result

    def discard_response(self):
        self.read_response()

    def pwm(self, duty_cycle: int):
        cmd = f'pwm {duty_cycle}\n'
        self.__device.write(cmd.encode())
        self.__device.flush()

    def on(self):
        self.__device.write('on\n'.encode())
        self.__device.flush()

    def off(self):
        self.__device.write('off\n'.encode())
        self.__device.flush()

    def pulse(self, delay: int = 0, duration_us: int = 50):
        cmd = f"pulse {delay} {duration_us}\n"
        self.__device.write(cmd.encode())
        self.__device.flush()

    def read_pulse_response(self) -> list[int]:
        timestamps: list[int] = []
        result = self.read_response()
        for r in result:
            timestamps.append(int(r))
        return timestamps


if __name__ == '__main__':
    # test program
    import time
    controller = PulseController()
    #controller.pulse(duration_us=1000)
    #response = controller.read_pulse_response()
    controller.pwm(127)
    response = controller.read_response()
    print(response)
