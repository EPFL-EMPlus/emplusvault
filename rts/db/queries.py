from sqlalchemy.sql import text
from rts.db.dao import DataAccessObject
from typing import Optional
from rts.api.models import LibraryBase, Projection, Media
import json


def get_library_id_from_name(library_name: str) -> Optional[int]:
    query = """
        SELECT library_id FROM library WHERE library_name=%s
    """
    library_id = DataAccessObject().fetch_one(query, (library_name,))
    return library_id['library_id'] if library_id else None


def create_library(library: LibraryBase):
    query = """
        INSERT INTO library (library_name, version, data)
        VALUES (%s, %s, %s)
        RETURNING library_id
    """
    vals = library.dict()
    library_id = DataAccessObject().execute_query(
        query, (vals['library_name'], vals['version'], json.dumps(vals['data'])))
    return {**library.dict(), "library_id": library_id.fetchone()[0]}


def create_projection(projection: Projection):
    projection_data = projection.dict()
    projection_data['model_params'] = json.dumps(
        projection_data['model_params'])
    projection_data['data'] = json.dumps(projection_data['data'])

    query = text("""
        INSERT INTO projection 
            (version, library_id, model_name, model_params, data, dimension, atlas_folder_path, atlas_width, 
            tile_size, atlas_count, total_tiles, tiles_per_atlas) 
        VALUES (:version, :library_id, :model_name, :model_params, :data, :dimension, :atlas_folder_path, 
            :atlas_width, :tile_size, :atlas_count, :total_tiles, :tiles_per_atlas)
    """)
    DataAccessObject().execute_query(query, projection_data)


def get_projection_by_id(projection_id: int):
    query = text(
        "SELECT * FROM projection WHERE projection_id = :projection_id")
    return DataAccessObject().fetch_one(
        query, {"projection_id": projection_id})


def get_all_projections():
    query = text("SELECT * FROM projection")
    return DataAccessObject().fetch_all(query)


def update_projection(projection_id: int, projection: Projection):
    projection_data = projection.dict()
    projection_data['model_params'] = json.dumps(
        projection_data['model_params'])
    projection_data['data'] = json.dumps(projection_data['data'])
    query = text("""
        UPDATE projection SET version = :version, library_id = :library_id, model_name = :model_name, model_params = :model_params, 
            data = :data, dimension = :dimension, atlas_folder_path = :atlas_folder_path, atlas_width = :atlas_width, 
            tile_size = :tile_size, atlas_count = :atlas_count, total_tiles = :total_tiles, 
            tiles_per_atlas = :tiles_per_atlas WHERE projection_id = :projection_id""")
    projection_data.update({"projection_id": projection_id})
    DataAccessObject().execute_query(query, projection_data)


def delete_projection(projection_id: int):
    query = text("DELETE FROM projection WHERE projection_id = :projection_id")
    DataAccessObject().execute_query(query, {"projection_id": projection_id})


def create_media(media: Media) -> dict:
    media_data = media.dict()
    media_data['metadata'] = json.dumps(media_data['metadata'])

    query = text("""
        INSERT INTO media (media_path, original_path, media_type, sub_type, size, metadata, library_id, hash, parent_id, start_ts, end_ts, start_frame, end_frame, frame_rate)
        VALUES (:media_path, :original_path, :media_type, :sub_type, :size, :metadata, :library_id, :hash, :parent_id, :start_ts, :end_ts, :start_frame, :end_frame, :frame_rate)
    """)
    DataAccessObject().execute_query(query, media_data)
    return media_data


def read_media_by_id(media_id: int) -> dict:
    query = text("SELECT * FROM media WHERE media_id = :media_id")
    return DataAccessObject().fetch_one(query, {"media_id": media_id})


def read_media() -> dict:
    query = text("SELECT * FROM media")
    return DataAccessObject().fetch_all(query)


def update_media(media_id: int, media: Media):
    media_data = media.dict()
    media_data['metadata'] = json.dumps(media_data['metadata'])

    query = text("""
        UPDATE media
        SET media_path=:media_path, original_path=:original_path, media_type=:media_type, sub_type=:sub_type, size=:size, metadata=:metadata, library_id=:library_id, hash=:hash, parent_id=:parent_id, start_ts=:start_ts, end_ts=:end_ts, start_frame=:start_frame, end_frame=:end_frame, frame_rate=:frame_rate
        WHERE media_id=:media_id
    """)
    DataAccessObject().execute_query(
        query, {**media_data, "media_id": media_id})
    return {**media_data, "media_id": media_id}


def delete_media(media_id: int):
    query = text("DELETE FROM media WHERE media_id = :media_id")
    DataAccessObject().execute_query(query, {"media_id": media_id})
    return {"status": "Media deleted"}
