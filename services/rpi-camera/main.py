import time
import struct
import asyncio
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from picamera2 import Picamera2
import uvicorn
from pulse import PulseController

X_RESOLUTION = 3280
Y_RESOLUTION = 2464

app = FastAPI()

# Enable CORS with custom headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['X-Image-Width', 'X-Image-Height']
)

# Initialize and configure camera
picam2 = Picamera2()
config = picam2.create_still_configuration(
    raw={'format': 'SRGGB10', 'size': (X_RESOLUTION, Y_RESOLUTION)})
picam2.configure(config)
picam2.start()
pulse = PulseController()


class CameraConfig(BaseModel):
    exposure: int
    gain: float


@app.get('/snapshot')
async def capture_raw_bayer(light_duty_cycle: int = 127):

    pulse.pwm(light_duty_cycle)
    pulse.discard_response()

    # Capture and discard a frame to ensure the frame
    # that we do get has been fully saturated at the
    # LED duty cycle we set.
    picam2.capture_array('raw')

    # This is the frame we keep.
    frame = picam2.capture_array('raw')

    pulse.off()
    pulse.discard_response()

    raw_bytes = frame.tobytes()

    headers = {
        # 16-bit Bayer, width halved
        'X-Image-Width': str(frame.shape[1] // 2),
        'X-Image-Height': str(frame.shape[0])
    }

    return Response(content=raw_bytes, media_type='application/octet-stream', headers=headers)


@app.get('/config')
async def get_config():
    metadata = picam2.capture_metadata()
    exposure = int(metadata.get('ExposureTime', 0))
    gain = float(metadata.get('AnalogueGain', 0.0))
    packed = struct.pack('<If', exposure, gain)
    return Response(content=packed, media_type='application/octet-stream')


@app.put('/config')
async def set_config(request: Request):
    try:
        body = await request.body()

        if len(body) != 8:
            raise ValueError(
                "Expected 8 bytes: 4 for gain (float32), 4 for exposure (uint32)")

        # Unpack LE binary: float32 gain, uint32 exposure
        exposure, gain = struct.unpack('<If', body)

        picam2.set_controls({
            "AnalogueGain": gain,
            "ExposureTime": exposure
        })

        return Response(content=b"ok", media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=6500)
