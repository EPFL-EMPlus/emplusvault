from pathlib import Path
from fastapi import (APIRouter, Depends, Request, HTTPException)
from fastapi.responses import FileResponse, Response
from typing import Any, Dict, Optional

# Local imports
from emv.api.api_settings import Settings, get_settings
from emv.utils import obj_from_json
from emv.db.queries import get_atlas, create_atlas, get_atlases, get_atlases_by_projection_id_and_order
from emv.settings import BUCKET_NAME
from emv.storage.storage import get_storage_client
from emv.api.models import Atlas
from emv.api.routers.auth_router import get_current_active_user, User

atlas_router = APIRouter()


def _get_atlas(atlas_name: str, settings: Settings) -> Optional[str]:
    atlas_folder = settings.get_atlases_folder()
    atlas_folder = Path(atlas_folder) / atlas_name
    if not atlas_folder.exists():
        return None
    atlas = atlas_folder / 'atlases.json'
    if not atlas.exists():
        return None
    return str(atlas)


@atlas_router.get("/images/{image_id}/{zoom}")
def get_image(req: Request, image_id: str, zoom: int, current_user: User = Depends(get_current_active_user)):

    sizes = {
        0: '32px',
        1: '64px',
        2: '128px',
        3: '256px',
        4: '512px',
        5: 'original'
    }
    size = sizes.get(zoom, 'original')

    image = get_storage_client().get_bytes(
        BUCKET_NAME, f'rts/images/{size}/{image_id}.jpg')

    return Response(content=image, media_type="image/jpeg")


@atlas_router.get("/atlases/{atlas_id}")
async def get_atlas_by_id(atlas_id: int, current_user: User = Depends(get_current_active_user)):
    result = get_atlas(atlas_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Atlas not found")
    return result


@atlas_router.get("/atlases/{atlas_id}/texture/{texture_id}")
async def get_atlas_image(atlas_id: int, texture_id: int, current_user: User = Depends(get_current_active_user)):

    atlas_entry = get_atlas(atlas_id)
    if atlas_entry is None:
        raise HTTPException(status_code=404, detail="Atlas not found")

    atlas = get_storage_client().get_bytes(
        BUCKET_NAME, atlas_entry['atlas_path'])

    return Response(content=atlas, media_type="image/png")

@atlas_router.get("/atlases/projection_id/{projection_id}/atlas_order/{atlas_order}")
async def get_atlas_by_projection_id_and_order(projection_id: int, atlas_order: int, current_user: User = Depends(get_current_active_user)):
    atlas_entry = get_atlases_by_projection_id_and_order(projection_id, atlas_order)[0]
    if atlas_entry is None:
        raise HTTPException(status_code=404, detail="Atlas not found")
    
    atlas = get_storage_client().get_bytes(
        BUCKET_NAME, atlas_entry['atlas_path'])
    return Response(content=atlas, media_type="image/png")

@atlas_router.post("/atlas/")
async def create(atlas: Atlas, current_user: User = Depends(get_current_active_user)):
    return create_atlas(atlas)


@atlas_router.get("/atlases/")
async def get_all_atlases(current_user: User = Depends(get_current_active_user)):
    return get_atlases()
