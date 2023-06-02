from pathlib import Path
from fastapi import (APIRouter, Depends, Request, HTTPException)
from fastapi.responses import FileResponse, Response
from typing import Any, Dict, Optional

# Local imports
from rts.api.settings import Settings, get_settings
from rts.utils import obj_from_json
from rts.db.queries import get_atlas, create_atlas, get_atlases
from rts.db_settings import BUCKET_NAME
from rts.storage.storage import get_supabase_client
from rts.api.models import AtlasCreate
from storage3.utils import StorageException

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


# @atlas_router.get("/atlases/{atlas_name}")
# def get_atlas(atlas_name: str = '0', settings: Settings = Depends(get_settings)):
#     atlas = _get_atlas(atlas_name, settings)
#     if not atlas:
#         raise HTTPException(
#             status_code=404, detail=f"Atlas not found {atlas_name}")

#     return FileResponse(atlas)


@atlas_router.get("/atlases/{atlas_id}")
async def get_atlas_by_id(atlas_id: int):
    result = get_atlas(atlas_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Atlas not found")
    return result


@atlas_router.get("/atlas_image/{atlas_id}")
async def get_atlas_image(atlas_id: int):

    atlas_entry = get_atlas(atlas_id)
    if atlas_entry is None:
        raise HTTPException(status_code=404, detail="Atlas not found")
    try:
        atlas = get_supabase_client().storage.from_(
            BUCKET_NAME).download(atlas_entry['atlas_path'])
    except StorageException as e:
        raise HTTPException(status_code=404, detail="Atlas not found")
    return Response(content=atlas, media_type="image/png")


@atlas_router.post("/atlas/")
async def create(atlas: AtlasCreate):
    return create_atlas(atlas)


@atlas_router.get("/atlases/")
async def get_all_atlases():
    return get_atlases()


# @atlas_router.get("/atlases/{atlas_name}/texture/{texture_id}")
# async def get_atlas(req: Request, atlas_name: str = '0', texture_id: int = 0, settings: Settings = Depends(get_settings)):
#     atlas = _get_atlas(atlas_name, settings)
#     if not atlas:
#         raise HTTPException(
#             status_code=404, detail=f"Atlas not found {atlas_name}")

#     atlases = obj_from_json(atlas)
#     try:
#         texture_path = atlases['atlases'][str(texture_id)]['fullpath']
#         return FileResponse(texture_path)
#     except:
#         raise HTTPException(
#             status_code=404, detail=f"Texture not found {texture_id}")
