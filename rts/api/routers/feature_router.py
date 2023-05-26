from fastapi import APIRouter, HTTPException
from sqlalchemy.sql import text
from typing import List
from rts.api.models import Feature
from rts.db.dao import DataAccessObject
from rts.db.queries import create_feature, read_feature_by_id, get_features, update_feature, delete_feature
import json


feature_router = APIRouter()


@feature_router.post("/feature/")
async def create(feature: Feature):
    return create_feature(feature)


@feature_router.get("/feature/{feature_id}", response_model=Feature)
async def read_feature(feature_id: int):
    result = read_feature_by_id(feature_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return result


@feature_router.get("/features/", response_model=List[Feature])
async def read_features():
    return get_features()


@feature_router.put("/feature/{feature_id}", response_model=Feature)
async def update(feature_id: int, feature: Feature):
    result = update_feature(feature_id, feature)
    if result is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return result


@feature_router.delete("/feature/{feature_id}", response_model=Feature)
async def delete(feature_id: int):
    result = delete_feature(feature_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return result
