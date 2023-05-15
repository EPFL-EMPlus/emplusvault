
import os
import logging
from pathlib import Path
from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from typing import Any, Dict, Tuple, Union, Optional

# Local imports
from rts.api.settings import Settings, get_settings, get_public_folder_path
from rts.utils import obj_from_json

# TODO: move the mount_routers function to a separate file

BYTES_PER_RESPONSE = 300000
stream_router = APIRouter()


def get_clip_id_from_image(image_id: str) -> str:
    toks = image_id.split('-')
    mediaId = toks[0]
    clip_number = f'{toks[1]}'
    clip_id = f'{mediaId}-{clip_number}'
    return clip_id


class StaticFilesSym(StaticFiles):
    "subclass StaticFiles middleware to allow symlinks"
    def lookup_path(self, path) -> Tuple[str, Any]:
        for directory in self.all_directories:
            full_path = os.path.realpath(os.path.join(directory, path))
            try:
                stat_result = os.stat(full_path)
                return (full_path, stat_result)
            except FileNotFoundError:
                pass
        return ("", None)


def get_index(request: Request, settings: Settings) -> HTMLResponse:
    templates = Jinja2Templates(directory=get_public_folder_path())
    base_url = ''
    if settings.host == 'localhost':
        base_url = 'http://' + settings.host + ':' + settings.app_port
        if settings.app_prefix:
            base_url += settings.app_prefix
    else:
        base_url = 'https://' + settings.host + settings.app_prefix
    return templates.TemplateResponse("index.html", {"request": request, "base_url": base_url})


@stream_router.get("/", response_class=HTMLResponse)
async def index(request: Request, settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return get_index(request, settings)


def mount_static_folder(app, route_name: str, folder_path: Union[str, Path]) -> None:
    app.mount(f"/{route_name}", StaticFilesSym(directory=Path(folder_path).expanduser()), name=route_name)


@stream_router.get("/images/{image_id}/{zoom}")
def get_image(req: Request, image_id: str, zoom: int):
    clip_id = get_clip_id_from_image(image_id)
    clip = req.app.state.clips.get(clip_id)

    if not clip:
        raise HTTPException(status_code=404, detail=f"Image not found {image_id}")

    sizes = {
        0: '32px',
        1: '64px',
        2: '128px',
        3: '256px',
        4: '512px',
        5: 'original'
    }

    size = sizes.get(zoom, 'original')
    image_path = Path(clip['clip_folder']) / 'images' / size / f'{image_id}.jpg'    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found {image_id}")
    
    return FileResponse(image_path)


def chunk_generator_from_stream(video_path: str, chunk_size: int, start: int, size: int):
    bytes_read = 0
    with open(video_path, 'rb') as stream:
        stream.seek(start)
        while bytes_read < size:
            bytes_to_read = min(chunk_size, size - bytes_read)
            yield stream.read(bytes_to_read)
            bytes_read += bytes_to_read


@stream_router.get('/stream/{image_id}')
async def stream_video(req: Request, image_id: str):
    clip_id = get_clip_id_from_image(image_id)
    clip = req.app.state.clips.get(clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail=f"Media not found {clip_id}")
    
    video_path = Path(clip['clip_folder']) / 'videos' / f'{clip_id}.mp4'
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Media not found {clip_id}")

    total_size = int(os.path.getsize(video_path))
    byte_ranges = [0, total_size]
    try:
        byte_ranges = req.headers.get("Range").split("=")[1].split('-')
    except AttributeError:
        pass

    start_byte_requested = int(byte_ranges[0])
    end_byte_requested = None
    if len(byte_ranges) > 1:  # safari
        if byte_ranges[1]:
            end_byte_requested = int(byte_ranges[1])

    # End byte is included, need to remove 1
    end_byte_planned = min(start_byte_requested + BYTES_PER_RESPONSE, total_size) - 1
    if end_byte_requested:
        end_byte_planned = min(end_byte_requested, end_byte_planned)

    size_chunk = min(BYTES_PER_RESPONSE, end_byte_planned + 1 - start_byte_requested)
    chunk_generator = chunk_generator_from_stream(
        video_path,
        chunk_size=BYTES_PER_RESPONSE,
        start=start_byte_requested,
        size=size_chunk
    )

    return StreamingResponse(
        chunk_generator,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start_byte_requested}-{end_byte_planned}/{total_size}",
            "Content-Type": 'video/mp4',
            "Content-Length": f"{end_byte_planned + 1 - start_byte_requested}"
        },
        status_code=206
    )    
