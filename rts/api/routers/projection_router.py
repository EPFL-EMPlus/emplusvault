from fastapi import (APIRouter, Depends, HTTPException)
from rts.api.models import Projection
from rts.utils import get_logger
from rts.db.queries import create_projection, get_projection_by_id, get_all_projections, update_projection, delete_projection, get_projection_coordinates, get_projection_atlases, get_projection_coordinates_by_atlas
from sqlalchemy.sql import text
from rts.api.routers.auth_router import get_current_active_user, User
import json

LOG = get_logger()

projection_router = APIRouter()
# TODO: should versions be auto incremented?


@projection_router.post("/projections/")
async def create(projection: Projection, current_user: User = Depends(get_current_active_user)):
    create_projection(projection)
    return {"status": "Projection created"}


@projection_router.get("/projections/")
async def read_projections(current_user: User = Depends(get_current_active_user)):
    return get_all_projections()


@projection_router.get("/projections/{projection_id}")
async def read_projection(projection_id: int, current_user: User = Depends(get_current_active_user)):
    projection = get_projection_by_id(projection_id)
    if projection:
        return projection
    raise HTTPException(status_code=404, detail="Projection not found")


@projection_router.get("/projections/{projection_id}/coordinates")
async def read_projection_coordinates(projection_id: int, current_user: User = Depends(get_current_active_user)):
    return get_projection_coordinates(projection_id)


@projection_router.get("/projections/{projection_id}/atlases")
async def read_projection_atlases(projection_id: int, current_user: User = Depends(get_current_active_user)):
    return get_projection_atlases(projection_id)


@projection_router.get("/projections/{projection_id}/coordinates/{atlas_id}")
async def read_projection_coordinates_by_atlas(projection_id: int, atlas_id: int, current_user: User = Depends(get_current_active_user)):
    return get_projection_coordinates_by_atlas(projection_id, atlas_id)


# @projection_router.put("/projections/{projection_id}")
# async def update(projection_id: int, projection: Projection, current_user: User = Depends(authenticate)):
#     update_projection(projection_id, projection)
#     return {"status": "Projection updated"}


# @projection_router.delete("/projections/{projection_id}")
# async def delete(projection_id: int, current_user: User = Depends(authenticate)):
#     delete_projection(projection_id)
#     return {"status": "Projection deleted"}


@projection_router.get("/coordinates/{projection_id}")
async def get_coordinates(projection_id: int, current_user: User = Depends(get_current_active_user)):
    return get_projection_coordinates(projection_id)
