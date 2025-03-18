from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.sql import text
from sqlalchemy.exc import IntegrityError, ProgrammingError
from emv.api.models import Media
from emv.api.routers.auth_router import get_current_active_user, User
from emv.db.dao import DataAccessObject
from emv.db.queries import create_media, get_media_by_id, get_all_media, create_or_update_media, delete_media, get_all_media_by_library_id, get_media_by_parent_id_and_types
import json

media_router = APIRouter()


@media_router.post("/media/")
async def create(media: Media, current_user: User = Depends(get_current_active_user)):
    try:
        media_data = create_or_update_media(media)
    except ProgrammingError as e:
        raise HTTPException(status_code=401, detail="Not allowed")
    return media_data


# Removed for now. With the amount of media files available, this call becomes too slow.
# @media_router.get("/media/", response_model=List[Media])
# async def get_medias(current_user: User = Depends(get_current_active_user)):
#     return get_all_media()


@media_router.get("/media/{media_id}")
async def get_media(media_id: str, current_user: User = Depends(get_current_active_user)):
    return get_media_by_id(media_id)


@media_router.get("/media/library/{library_id}")
async def get_media_by_library(library_id: int, last_seen_date: str = None, last_seen_media_id: str = None, page_size: int = 20, media_type: str = None, sub_type: str = None, current_user: User = Depends(get_current_active_user)):
    resp = get_all_media_by_library_id(
        library_id, last_seen_date, last_seen_media_id, page_size, media_type, sub_type)
    return resp


@media_router.get("/media/library/{library_id}/parent/{parent_id}/type/{media_type}/sub_type/{sub_type}")
async def get_media_by_library_and_parent(library_id: int, parent_id: int, media_type: str, sub_type: str, current_user: User = Depends(get_current_active_user)):
    return get_media_by_parent_id_and_types(parent_id, media_type, sub_type)


# @media_router.put("/media/{media_id}")
# async def update(media_id: int, media: Media, supabase: Client = Depends(authenticate)):
#     return update_media(media_id, media)
