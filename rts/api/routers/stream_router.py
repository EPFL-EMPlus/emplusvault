from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from io import BytesIO
from rts.api.routers.auth_router import get_current_active_user, User
from rts.db.queries import log_access, get_media_for_streaming
from rts.storage.storage import get_storage_client
from rts.settings import BUCKET_NAME

BYTES_PER_RESPONSE = 300000
stream_router = APIRouter()


@stream_router.get('/stream/{media_id}')
async def stream_video(req: Request, media_id: str, current_user: User = Depends(get_current_active_user)):

    # log_access(current_user, media_id)
    media = get_media_for_streaming(media_id)

    total_size = media['size']
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

    stream = get_storage_client().get_stream(
        media['library_name'], media['media_path'],
        start=start_byte_requested, end=end_byte_planned)

    return StreamingResponse(
        stream,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start_byte_requested}-{end_byte_planned}/{total_size}",
            "Content-Type": 'video/mp4',
            "Content-Length": f"{end_byte_planned + 1 - start_byte_requested}"
        },
        status_code=206
    )
