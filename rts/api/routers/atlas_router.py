from pathlib import Path
from fastapi import (APIRouter, Depends, Request, HTTPException)
from fastapi.responses import FileResponse, Response
from typing import Any, Dict, Optional

# Local imports
from rts.api.settings import Settings, get_settings
from rts.utils import obj_from_json
from rts.db.queries import get_atlas, create_atlas, get_atlases
from rts.db_settings import BUCKET_NAME
from rts.storage.storage import get_storage_client
from rts.api.models import AtlasCreate
from storage3.utils import StorageException
from rts.api.routers.auth_router import authenticate
from supabase import Client

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
def get_image(req: Request, image_id: str, zoom: int, supabase: Client = Depends(authenticate)):

    sizes = {
        0: '32px',
        1: '64px',
        2: '128px',
        3: '256px',
        4: '512px',
        5: 'original'
    }
    size = sizes.get(zoom, 'original')

    try:
        image = get_storage_client().download(
            BUCKET_NAME, f'rts/images/{size}/{image_id}.jpg')
    except StorageException as e:
        raise HTTPException(status_code=404, detail="Atlas not found")
    return Response(content=image, media_type="image/jpeg")


@atlas_router.get("/atlases/{atlas_id}")
async def get_atlas_by_id(atlas_id: int, supabase: Client = Depends(authenticate)):
    result = get_atlas(atlas_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Atlas not found")
    return result


@atlas_router.get("/atlases/{atlas_id}/texture/{texture_id}")
async def get_atlas_image(atlas_id: int, texture_id: int, supabase: Client = Depends(authenticate)):

    atlas_entry = get_atlas(atlas_id)
    if atlas_entry is None:
        raise HTTPException(status_code=404, detail="Atlas not found")
    try:
        atlas = get_storage_client().download(
            BUCKET_NAME, atlas_entry['atlas_path'])
    except StorageException as e:
        raise HTTPException(status_code=404, detail="Atlas not found")
    return Response(content=atlas, media_type="image/png")


@atlas_router.post("/atlas/")
async def create(atlas: AtlasCreate, supabase: Client = Depends(authenticate)):
    return create_atlas(atlas)


@atlas_router.get("/atlases/")
async def get_all_atlases(supabase: Client = Depends(authenticate)):
    return get_atlases()
