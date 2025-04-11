from fastapi import FastAPI, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import threading

from app.server import Server

server = Server()
server_lock = threading.Lock()
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post('/api/reset')
def reset(report_id: str = Query(..., description='Patient/Report ID'),
          timestamp: str = Query(...,
                                 description='The timestamp to attach to the report.'),
          notes: str = Query('', description='Any additional notes to attach to the report.')):
    with server_lock:
        return server.reset(report_id, timestamp, notes)


@app.post('/api/update')
def update(image_bytes: bytes = Body(..., description='Raw JPG/PNG/BMP image data')):
    with server_lock:
        return server.update(image_bytes)


@app.post('/api/finalize')
def finalize():
    with server_lock:
        return server.finalize()


@app.get("/api/report")
def get_latest_report():
    with server_lock:
        # A lock might not actually be needed for this call, but just in case things change.
        return JSONResponse(server.report())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
