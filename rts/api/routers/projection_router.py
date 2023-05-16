from fastapi import (APIRouter, Depends, HTTPException)
from rts.api.models import Projection
from rts.utils import get_logger
from rts.db.queries import create_projection, get_projection_by_id, get_all_projections, update_projection, delete_projection
from sqlalchemy.sql import text
import json

LOG = get_logger()

projection_router = APIRouter()
# TODO: should versions be auto incremented?


@projection_router.post("/projections/")
async def create(projection: Projection):
    create_projection(projection)
    return {"status": "Projection created"}


@projection_router.get("/projections/")
async def read_projections():
    return get_all_projections()


@projection_router.get("/projections/{projection_id}")
async def read_projection(projection_id: int):
    projection = get_projection_by_id(projection_id)
    if projection:
        return projection
    raise HTTPException(status_code=404, detail="Projection not found")


@projection_router.put("/projections/{projection_id}")
async def update(projection_id: int, projection: Projection):
    update_projection(projection_id, projection)
    return {"status": "Projection updated"}


@projection_router.delete("/projections/{projection_id}")
async def delete(projection_id: int):
    delete_projection(projection_id)
    return {"status": "Projection deleted"}
