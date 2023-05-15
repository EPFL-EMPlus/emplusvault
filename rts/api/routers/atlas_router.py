from pathlib import Path
from fastapi import (APIRouter, Depends, Request, HTTPException)
from fastapi.responses import FileResponse
from typing import Any, Dict, Optional

# Local imports
from rts.api.settings import Settings, get_settings
from rts.utils import obj_from_json

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


@atlas_router.get("/atlases/{atlas_name}")
def get_atlas(atlas_name: str = '0', settings: Settings = Depends(get_settings)):
    atlas = _get_atlas(atlas_name, settings)
    if not atlas:
        raise HTTPException(status_code=404, detail=f"Atlas not found {atlas_name}")

    return FileResponse(atlas)


@atlas_router.get("/atlases/{atlas_name}/texture/{texture_id}")
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
