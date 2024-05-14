from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.sql import text
from typing import List
from emv.api.models import Feature
from emv.db.dao import DataAccessObject
from emv.db.queries import create_feature, get_feature_by_id, get_all_features, update_feature, delete_feature, get_features_by_type_paginated
from emv.features.pose import get_keypoints_from_image
import json
from datetime import datetime
from emv.api.routers.auth_router import get_current_active_user, User
from PIL import Image
from io import BytesIO

feature_router = APIRouter()


@feature_router.post("/feature/")
async def create(feature: Feature, current_user: User = Depends(get_current_active_user)):
    if feature.created_at is None:
        feature.created_at = datetime.now()
    return create_feature(feature)


@feature_router.get("/feature/{feature_id}", response_model=Feature)
async def read_feature(feature_id: int, current_user: User = Depends(get_current_active_user)):
    result = get_feature_by_id(feature_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Feature not found")
    return result


# @feature_router.get("/features/", response_model=List[Feature])
# async def read_features(current_user: User = Depends(get_current_active_user)):
#     return get_all_features()

@feature_router.get("/features/{feature_type}")
async def get_media_by_library(feature_type: str, page_size: int = 20, last_seen_feature_id: int = -1, current_user: User = Depends(get_current_active_user)):
    try:
        resp = get_features_by_type_paginated(
            feature_type, page_size, last_seen_feature_id)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Not allowed")
    return resp

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


# get keypoints from uploaded image
@feature_router.post("/feature/keypoints/")
async def get_keypoints(file: UploadFile = File(...)):

    if not file.content_type.startswith('image/'):
        return JSONResponse(status_code=400, content={"message": "File is not an image"})

    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))

        # get keypoints from ml model
        keypoints = get_keypoints_from_image(image)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Error processing image"})
    
    return keypoints
