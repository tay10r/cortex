import serial


class PulseController:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.__device = serial.Serial(port, baudrate, timeout=1)

    def pulse(self, duration_us: int = 50) -> list[int]:
        cmd = f"{duration_us}\n"
        self.__device.write(cmd.encode())
        self.__device.flush()
        timestamps: list[int] = []
        while True:
            line = self.__device.readline()
            if line == b'':
                break
            data = line.decode().strip()
            if data == '':
                break
            t = int(data)
            timestamps.append(t)
        return timestamps


if __name__ == '__main__':
    # test program
    import time
    controller = PulseController()
    response = controller.pulse(duration_us=1000)
    print(response)
