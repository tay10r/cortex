from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from picamera2 import Picamera2
import time
import uvicorn

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
config = picam2.create_still_configuration(raw={'format': 'SRGGB10', 'size': (X_RESOLUTION, Y_RESOLUTION)})
picam2.configure(config)
picam2.start()

# Warm up
time.sleep(1)

class CameraConfig(BaseModel):
    exposure: float | None = None  # in microseconds
    gain: float | None = None      # analog gain

@app.get('/snapshot')
async def capture_raw_bayer():
    frame = picam2.capture_array('raw')
    raw_bytes = frame.tobytes()

    headers = {
        'X-Image-Width': str(frame.shape[1] // 2),  # 16-bit Bayer, width halved
        'X-Image-Height': str(frame.shape[0])
    }

    return Response(content=raw_bytes, media_type='application/octet-stream', headers=headers)

@app.get('/config')
async def get_config():
    metadata = picam2.capture_metadata()
    return {
        'exposure': metadata.get('ExposureTime', None),
        'gain': metadata.get('AnalogueGain', None)
    }

@app.post('/config')
async def set_config(config: CameraConfig):
    controls = {}

    if config.exposure is not None:
        controls['ExposureTime'] = int(config.exposure)  # microseconds
    if config.gain is not None:
        controls['AnalogueGain'] = float(config.gain)

    if not controls:
        raise HTTPException(status_code=400, detail='No valid parameters provided.')

    try:
        picam2.set_controls(controls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {}

if __name__ == '__main__':
    uvicorn.run('your_filename:app', host='0.0.0.0', port=6500)
