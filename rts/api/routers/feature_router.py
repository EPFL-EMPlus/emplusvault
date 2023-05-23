from fastapi import APIRouter, HTTPException
from sqlalchemy.sql import text
from typing import List
from rts.api.models import Feature
from rts.db.dao import DataAccessObject
import json
feature_router = APIRouter()


@feature_router.post("/feature/")
async def create_feature(feature: Feature):
    query = text("""
        INSERT INTO feature (feature_type, version, model_name, model_params, data, 
        embedding_size, embedding_1024, embedding_1536, embedding_2048, media_id)
        VALUES (:feature_type, :version, :model_name, :model_params, :data, 
        :embedding_size, :embedding_1024, :embedding_1536, :embedding_2048, :media_id)
        RETURNING feature_id
    """)
    feature_dict = feature.dict()
    feature_dict["model_params"] = json.dumps(feature_dict["model_params"])
    feature_dict["data"] = json.dumps(feature_dict["data"])
    result = DataAccessObject().execute_query(query, feature_dict)
    return {**feature_dict, "feature_id": result.fetchone()[0]}


@feature_router.get("/feature/{feature_id}", response_model=Feature)
async def read_feature(feature_id: int):
    query = text("SELECT * FROM feature WHERE feature_id = :feature_id")
    result = DataAccessObject().fetch_one(query, {"feature_id": feature_id})
    if result is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return result


@feature_router.get("/features/", response_model=List[Feature])
async def read_features():
    query = text("SELECT * FROM feature")
    result = DataAccessObject().fetch_all(query)
    return result


@feature_router.put("/feature/{feature_id}", response_model=Feature)
async def update_feature(feature_id: int, feature: Feature):
    feature_dict = feature.dict()
    feature_dict["model_params"] = json.dumps(feature_dict["model_params"])
    feature_dict["data"] = json.dumps(feature_dict["data"])
    query = text("""
        UPDATE feature
        SET feature_type = :feature_type, version = :version, model_name = :model_name, 
        model_params = :model_params, data = :data, embedding_size = :embedding_size, 
        embedding_1024 = :embedding_1024, embedding_1536 = :embedding_1536, 
        embedding_2048 = :embedding_2048, media_id = :media_id
        WHERE feature_id = :feature_id
        RETURNING *
    """)
    result = DataAccessObject().fetch_one(
        query, {**feature_dict, "feature_id": feature_id})
    if result is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return result


@feature_router.delete("/feature/{feature_id}", response_model=Feature)
async def delete_feature(feature_id: int):
    query = text("""
        DELETE FROM feature
        WHERE feature_id = :feature_id
        RETURNING *
    """)
    result = DataAccessObject().fetch_one(query, {"feature_id": feature_id})
    if result is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return result
