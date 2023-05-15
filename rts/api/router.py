
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
router = APIRouter()


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


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return get_index(request, settings)


def mount_routers(app, settings: Settings) -> None:
    app.include_router(router)
    if settings.mode == 'prod':
        mount_point = settings.app_prefix if settings.app_prefix else ''
        # print("MOUNT POINT: ", mount_point)
        app.mount(mount_point + "/", StaticFiles(directory=get_public_folder_path(), html=True), name="public")
    else:
        app.mount("/", StaticFiles(directory=get_public_folder_path(), html=True), name="public")
        

def mount_static_folder(app, route_name: str, folder_path: Union[str, Path]) -> None:
    app.mount(f"/{route_name}", StaticFilesSym(directory=Path(folder_path).expanduser()), name=route_name)


@router.get("/images/{image_id}/{zoom}")
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


def _get_atlas(atlas_name: str, settings: Settings) -> Optional[str]:
    atlas_folder = settings.get_atlases_folder()
    atlas_folder = Path(atlas_folder) / atlas_name
    if not atlas_folder.exists():
        return None
    atlas = atlas_folder / 'atlases.json'
    if not atlas.exists():
        return None
    return str(atlas)


@router.get("/atlases/{atlas_name}")
def get_atlas(atlas_name: str = '0', settings: Settings = Depends(get_settings)):
    atlas = _get_atlas(atlas_name, settings)
    if not atlas:
        raise HTTPException(status_code=404, detail=f"Atlas not found {atlas_name}")

    return FileResponse(atlas)


@router.get("/atlases/{atlas_name}/texture/{texture_id}")
def get_atlas(req: Request, atlas_name: str = '0', texture_id: int = 0, settings: Settings = Depends(get_settings)):
    atlas = _get_atlas(atlas_name, settings)
    if not atlas:
        raise HTTPException(status_code=404, detail=f"Atlas not found {atlas_name}")
    
    atlases = obj_from_json(atlas)
    try:
        texture_path = atlases['atlases'][str(texture_id)]['fullpath']
        return FileResponse(texture_path)
    except:
        raise HTTPException(status_code=404, detail=f"Texture not found {texture_id}")


def chunk_generator_from_stream(video_path: str, chunk_size: int, start: int, size: int):
    bytes_read = 0
    with open(video_path, 'rb') as stream:
        stream.seek(start)
        while bytes_read < size:
            bytes_to_read = min(chunk_size, size - bytes_read)
            yield stream.read(bytes_to_read)
            bytes_read += bytes_to_read


@router.get('/stream/{image_id}')
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
