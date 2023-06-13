from pydantic import BaseModel, Field, Json
from datetime import datetime
from typing import Dict, Optional, List, Tuple


class LibraryBase(BaseModel):
    library_name: str
    version: str
    data: Json


class LibraryCreate(LibraryBase):
    data: str


class Library(LibraryBase):
    library_id: int
    created_at: datetime


class Projection(BaseModel):
    projection_id: Optional[int] = Field(None, alias="projection_id")
    version: str = Field(..., alias="version", max_length=20)
    library_id: int = Field(..., alias="library_id")
    created_at: Optional[datetime] = Field(None, alias="created_at")
    model_name: str = Field(..., alias="model_name", max_length=200)
    model_params: Dict = Field(..., alias="model_params")
    data: Dict = Field(..., alias="data")
    dimension: int = Field(..., alias="dimension")
    atlas_folder_path: str = Field(...,
                                   alias="atlas_folder_path", max_length=500)
    atlas_width: int = Field(..., alias="atlas_width")
    tile_size: int = Field(..., alias="tile_size")
    atlas_count: int = Field(..., alias="atlas_count")
    total_tiles: int = Field(..., alias="total_tiles")
    tiles_per_atlas: int = Field(..., alias="tiles_per_atlas")


class Media(BaseModel):
    media_id: Optional[int] = Field(None, alias="media_id")
    media_path: str = Field(..., alias="media_path", max_length=500)
    original_path: str = Field(..., alias="original_path", max_length=500)
    created_at: Optional[datetime] = Field(None, alias="created_at")
    media_type: str = Field(..., alias="media_type", max_length=50)
    sub_type: str = Field(..., alias="sub_type", max_length=50)
    size: int = Field(..., alias="size")
    metadata: Dict = Field(..., alias="metadata")
    library_id: int = Field(..., alias="library_id")
    hash: Optional[str] = Field(None, alias="hash", max_length=50)
    parent_id: Optional[int] = Field(None, alias="parent_id")
    start_ts: Optional[float] = Field(None, alias="start_ts")
    end_ts: Optional[float] = Field(None, alias="end_ts")
    start_frame: Optional[int] = Field(None, alias="start_frame")
    end_frame: Optional[int] = Field(None, alias="end_frame")
    frame_rate: Optional[float] = Field(None, alias="frame_rate")


class Feature(BaseModel):
    feature_id: Optional[int]
    feature_type: str
    version: str
    created_at: Optional[datetime]
    model_name: str
    model_params: Dict
    data: Dict

    embedding_size: Optional[int]
    embedding_1024: Optional[List[float]]
    embedding_1536: Optional[List[float]]
    embedding_2048: Optional[List[float]]

    media_id: int


class MapProjectionFeatureBase(BaseModel):
    projection_id: int
    atlas_order: int
    # Represented as (latitude, longitude) or (x, y)
    coordinates: Tuple[float, float]
    index_in_atlas: int


class MapProjectionFeatureCreate(MapProjectionFeatureBase):
    feature_id: Optional[int] = None
    media_id: Optional[int] = None


class MapProjectionFeature(MapProjectionFeatureBase):
    map_projection_feature_id: int

    class Config:
        orm_mode = True


class AtlasBase(BaseModel):
    projection_id: int
    atlas_order: int
    atlas_path: str
    atlas_size: List[float]
    tile_size: List[float]
    tile_count: int
    rows: int
    cols: int
    tiles_per_atlas: int


class AtlasCreate(AtlasBase):
    pass


class Atlas(AtlasBase):
    atlas_id: int

    class Config:
        orm_mode = True
