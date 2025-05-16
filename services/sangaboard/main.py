from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import serial

SERIAL_PORT = '/dev/ttyAMA0'
BAUD_RATE = 115200
TIMEOUT = 1  # seconds

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post('/command', response_class=Response, status_code=200)
async def send_command(request: Request):
    # Read raw body as bytes, decode to str
    command = await request.body()
    command_str = command.decode().strip()

    if not command_str:
        return Response('Empty command', status_code=400)

    # Send command to the serial device
    ser.write((command_str + '\n').encode())

    # Read response line
    response_line = ser.readline().decode(errors='replace').strip()

    return Response(content=response_line, media_type='text/plain')
