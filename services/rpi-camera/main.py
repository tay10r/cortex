from flask import Flask, Response
from flask_cors import CORS
from picamera2 import Picamera2
import time

X_RESOLUTION = 3820
Y_RESOLUTION = 2464

app = Flask(__name__)
CORS(app, expose_headers=['X-Image-Width', 'X-Image-Height'])

picam2 = Picamera2()

# Configure camera for raw Bayer capture only
config = picam2.create_still_configuration(raw={'format': 'SRGGB10', 'size': (X_RESOLUTION, Y_RESOLUTION)})
picam2.configure(config)
picam2.start()

# Warm up camera
time.sleep(1)

@app.route('/snapshot', methods=['GET'])
def capture_raw_bayer():
    # Capture raw Bayer frame only (no post-processing)
    frame = picam2.capture_array('raw')

    # Convert to bytes
    raw_bytes = frame.tobytes()
    response = Response(raw_bytes, content_type='application/octet-stream')
    response.headers['X-Image-Width'] = str(frame.shape[1] // 2) # divide by 2 since this is 16 bit data
    response.headers['X-Image-Height'] = str(frame.shape[0])
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6500)
