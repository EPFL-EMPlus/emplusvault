from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.sql import text
from typing import List
from rts.api.models import Feature
from rts.db.dao import DataAccessObject
from rts.db.queries import create_feature, read_feature_by_id, get_features, update_feature, delete_feature
import json
from rts.api.routers.auth_router import authenticate
from supabase import Client

feature_router = APIRouter()


@feature_router.post("/feature/")
async def create(feature: Feature, supabase: Client = Depends(authenticate)):
    return create_feature(feature)


@feature_router.get("/feature/{feature_id}", response_model=Feature)
async def read_feature(feature_id: int, supabase: Client = Depends(authenticate)):
    result = read_feature_by_id(feature_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return result


@feature_router.get("/features/", response_model=List[Feature])
async def read_features(supabase: Client = Depends(authenticate)):
    return get_features()


# @feature_router.put("/feature/{feature_id}", response_model=Feature)
# async def update(feature_id: int, feature: Feature, supabase: Client = Depends(authenticate)):
#     result = update_feature(feature_id, feature)
#     if result is None:
#         raise HTTPException(status_code=404, detail="Feature not found")
#     return result


# @feature_router.delete("/feature/{feature_id}", response_model=Feature)
# async def delete(feature_id: int, supabase: Client = Depends(authenticate)):
#     result = delete_feature(feature_id)
#     if result is None:
#         raise HTTPException(status_code=404, detail="Feature not found")
#     return result
