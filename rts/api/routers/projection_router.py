from fastapi import (APIRouter, Depends, HTTPException)
from rts.db.dao import DataAccessObject
from rts.api.models import Projection
from rts.utils import get_logger
from sqlalchemy.sql import text
import json

LOG = get_logger()

projection_router = APIRouter() 
# TODO: should versions be auto incremented?

@projection_router.post("/projections/")
async def create_projection(projection: Projection):
    projection_data = projection.dict()
    projection_data['model_params'] = json.dumps(projection_data['model_params'])
    projection_data['data'] = json.dumps(projection_data['data'])

    query = text("""
        INSERT INTO projection 
            (version, library_id, model_name, model_params, data, dimension, atlas_folder_path, atlas_width, 
            tile_size, atlas_count, total_tiles, tiles_per_atlas) 
        VALUES (:version, :library_id, :model_name, :model_params, :data, :dimension, :atlas_folder_path, 
            :atlas_width, :tile_size, :atlas_count, :total_tiles, :tiles_per_atlas)
    """)
    DataAccessObject().execute_query(query, projection_data)

    return {"status": "Projection created"}


@projection_router.get("/projections/")
async def read_projections():
    query = text("SELECT * FROM projection")
    return DataAccessObject().fetch_all(query)

@projection_router.get("/projections/{projection_id}")
async def read_projection(projection_id: int):
    query = text("SELECT * FROM projection WHERE projection_id = :projection_id")
    projection = DataAccessObject().fetch_one(query, {"projection_id": projection_id})
    if projection:
        return projection
    raise HTTPException(status_code=404, detail="Projection not found")

@projection_router.put("/projections/{projection_id}")
async def update_projection(projection_id: int, projection: Projection):
    projection_data = projection.dict()
    projection_data['model_params'] = json.dumps(projection_data['model_params'])
    projection_data['data'] = json.dumps(projection_data['data'])
    query = text("""
        UPDATE projection SET version = :version, library_id = :library_id, model_name = :model_name, model_params = :model_params, 
            data = :data, dimension = :dimension, atlas_folder_path = :atlas_folder_path, atlas_width = :atlas_width, 
            tile_size = :tile_size, atlas_count = :atlas_count, total_tiles = :total_tiles, 
            tiles_per_atlas = :tiles_per_atlas WHERE projection_id = :projection_id""")
    projection_data.update({"projection_id": projection_id})
    DataAccessObject().execute_query(query, projection_data)
    return {"status": "Projection updated"}

@projection_router.delete("/projections/{projection_id}")
async def delete_projection(projection_id: int):
    query = text("DELETE FROM projection WHERE projection_id = :projection_id")
    DataAccessObject().execute_query(query, {"projection_id": projection_id})
    return {"status": "Projection deleted"}
