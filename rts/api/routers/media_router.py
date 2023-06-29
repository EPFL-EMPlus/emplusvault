from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.sql import text
from rts.api.models import Media
from rts.api.routers.auth_router import get_current_active_user
from supabase import Client
from rts.db.dao import DataAccessObject
from rts.db.queries import create_media, read_media_by_id, read_media, update_media, delete_media
import json

media_router = APIRouter()


@media_router.post("/media/")
async def create(media: Media, supabase: Client = Depends(get_current_active_user)):
    media_data = create_media(media)
    return media_data


@media_router.get("/media/", response_model=List[Media])
async def get_medias(supabase: Client = Depends(get_current_active_user)):
    return read_media()


@media_router.get("/media/{media_id}", response_model=Media)
async def get_media(media_id: int, supabase: Client = Depends(get_current_active_user)):
    return read_media_by_id(media_id)


# @media_router.put("/media/{media_id}")
# async def update(media_id: int, media: Media, supabase: Client = Depends(authenticate)):
#     return update_media(media_id, media)
