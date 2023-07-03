from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from io import BytesIO
from rts.api.routers.auth_router import get_current_active_user, User
from rts.storage.storage import get_storage_client
from rts.db_settings import BUCKET_NAME

BYTES_PER_RESPONSE = 300000
stream_router = APIRouter()


def chunk_generator_from_stream(stream, chunk_size: int, start: int, size: int):
    bytes_read = 0
    stream.seek(start)
    while bytes_read < size:
        bytes_to_read = min(chunk_size, size - bytes_read)
        yield stream.read(bytes_to_read)
        bytes_read += bytes_to_read


def get_clip_id_from_image(image_id: str) -> str:
    toks = image_id.split('-')
    mediaId = toks[0]
    clip_number = f'{toks[1]}'
    clip_id = f'{mediaId}-{clip_number}'
    return clip_id


@stream_router.get('/stream/{image_id}')
async def stream_video(req: Request, image_id: str, current_user: User = Depends(get_current_active_user)):

    # Get bucket ID from the database

    video_name = "ZB001020-L000.mp4"

    r = get_storage_client().download(
        BUCKET_NAME, f"{BUCKET_NAME}/videos/{video_name}")
    stream_video = BytesIO(r)

    total_size = stream_video.getbuffer().nbytes
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
    end_byte_planned = min(start_byte_requested +
                           BYTES_PER_RESPONSE, total_size) - 1
    if end_byte_requested:
        end_byte_planned = min(end_byte_requested, end_byte_planned)

    size_chunk = min(BYTES_PER_RESPONSE, end_byte_planned +
                     1 - start_byte_requested)

    chunk_generator = chunk_generator_from_stream(
        stream_video,
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
