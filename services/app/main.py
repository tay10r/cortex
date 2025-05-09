#!/usr/bin/env python3

# This is a web server that hosts:
#  - web page files for the user interface
#  - endpoints for storing, retrieving, and deleting images
#  - an endpoint for invoking the model

from pathlib import Path
import uuid
import json
import time
from dataclasses import dataclass, asdict
from asyncio import Lock
import asyncio
import aiofiles
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
import uvicorn


@dataclass
class Config:
    static_dir: str = 'static'
    images_dir: str = 'images'
    images_index: str = 'images.json'
    host_ip: str = '0.0.0.0'
    host_port: int = 8000


@dataclass
class ImageInfo:
    width: int
    height: int
    creation_time: int


class Server:
    def __init__(self, config: Config):
        self.__config = config
        self.__images_index: dict[str, dict] = {}
        self.__index_lock = Lock()
        self.__load_images_index()

    def get_images_index(self) -> dict:
        return self.__images_index

    def get_image(self, image_id: str) -> FileResponse:
        # sanitize ID
        image_id = Server.__safe_uuid(image_id)
        path = Path(self.__config.images_dir) / f'{image_id}.bin'
        if not path.exists():
            raise HTTPException(404, 'Image not found')
        return FileResponse(path)

    async def upload(self, image_data: bytes, width: int, height: int) -> dict:
        image_id = str(uuid.uuid4())
        image_path = Path(self.__config.images_dir) / f'{image_id}.bin'
        async with aiofiles.open(image_path, 'wb') as f:
            await f.write(image_data)
        image_info = ImageInfo(width=width,
                               height=height,
                               creation_time=int(time.time()))
        async with self.__index_lock:
            self.__images_index[image_id] = asdict(image_info)
        await self.__save_images_index()
        return {'image_id': image_id}

    async def delete(self, image_id: str) -> dict:
        # normalize UUID
        image_id = Server.__safe_uuid(image_id)
        image_path = Path(self.__config.images_dir) / f"{image_id}.bin"
        if not image_path.exists():
            raise HTTPException(404, "Image not found")
        await asyncio.to_thread(image_path.unlink)
        # update index
        async with self.__index_lock:
            self.__images_index.pop(image_id, None)
        await self.__save_images_index()
        return {}

    @staticmethod
    def __safe_uuid(image_id: str) -> str:
        try:
            return str(uuid.UUID(image_id))
        except ValueError:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid image ID")

    def __load_images_index(self):
        path = Path(self.__config.images_index)
        if not path.exists():
            return
        with open(path, 'r') as f:
            self.__images_index = json.load(f)

    async def __save_images_index(self):
        path = Path(self.__config.images_index)
        async with aiofiles.open(path, 'w') as f:
            await f.write(json.dumps(self.__images_index))


def open_config(filename: str = 'config.json') -> Config:
    p = Path(filename)
    config = Config()
    if p.is_file():
        with open(p, 'r') as f:
            config = Config(**json.load(f))
    return config


def create_app(server: Server, config: Config) -> FastAPI:
    app = FastAPI()
    app.mount('/static',
              StaticFiles(directory=config.static_dir, html=True),
              name='static')

    @app.get('/images.json')
    async def list_images() -> dict:
        return server.get_images_index()

    @app.get('/images/{image_id}.bin')
    async def get_image(image_id: str) -> FileResponse:
        return server.get_image(image_id)

    @app.post('/upload')
    async def upload(request: Request, width: int = Query(..., gt=0), height: int = Query(..., gt=0)) -> dict:
        image_data = await request.body()
        return await server.upload(image_data, width, height)

    @app.delete("/images/{image_id}.bin")
    async def delete_image(image_id: str) -> dict:
        return await server.delete(image_id)

    return app


def main():
    config = open_config()
    Path(config.images_dir).mkdir(exist_ok=True)
    server = Server(config)
    app = create_app(server, config)
    uvicorn.run(app, host=config.host_ip, port=config.host_port)


if __name__ == '__main__':
    main()
