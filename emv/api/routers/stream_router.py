from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, RedirectResponse
from io import BytesIO
from emv.api.routers.auth_router import get_current_active_user, User
from emv.db.queries import check_access, get_media_for_streaming
from emv.storage.storage import get_storage_client
from emv.settings import BUCKET_NAME
from fastapi.responses import JSONResponse

BYTES_PER_RESPONSE = 5_000_000
stream_router = APIRouter()


@stream_router.get('/stream/{media_id}')
async def stream_video4(req: Request, media_id: str, current_user: User = Depends(get_current_active_user)):
    check_access(current_user, media_id)
    media = get_media_for_streaming(media_id)
    url = get_storage_client().generate_presigned_url(
        media['library_name'], media['media_path'])
    return RedirectResponse(url=url)


@stream_router.get('/redirection_url/{media_id}')
async def redirection_url(media_id: str, current_user: User = Depends(get_current_active_user)):
    check_access(current_user, media_id)
    media = get_media_for_streaming(media_id)
    url = get_storage_client().generate_presigned_url(
        media['library_name'], media['media_path'])
    return JSONResponse(content={"url": url})


@stream_router.get('/download/{media_id}')
def download_video(media_id: str, current_user: User = Depends(get_current_active_user)):
    check_access(current_user, media_id)
    media = get_media_for_streaming(media_id)
    stream = get_storage_client().get_bytes(
        media['library_name'], media['media_path'])
    return StreamingResponse(
        BytesIO(stream),
        headers={
            "Content-Type": 'video/mp4',
            "Content-Disposition": f'attachment; filename="{media_id}.mp4"'
        }
    )


@stream_router.get('/hls/{media_id}/playlist.m3u8')
async def get_playlist(media_id: str, current_user: User = Depends(get_current_active_user)):
    stream = get_storage_client().get_bytes("rts", "test/output.m3u8")
    return StreamingResponse(BytesIO(stream), media_type="application/x-mpegURL")


@stream_router.get('/hls/{media_id}/{segment_file}')
async def get_segment(media_id: str, segment_file: str, current_user: User = Depends(get_current_active_user)):
    stream = get_storage_client().get_bytes("rts", f"test/{segment_file}")
    return StreamingResponse(BytesIO(stream), media_type="video/MP2T")
