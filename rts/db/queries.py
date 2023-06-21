from sqlalchemy.sql import text
from rts.db.dao import DataAccessObject
from typing import Optional
from rts.api.models import LibraryBase, Projection, Media, Feature, MapProjectionFeatureCreate, AtlasCreate
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
        RETURNING projection_id
    """)
    projection_id = DataAccessObject().execute_query(query, projection_data)
    return {**projection.dict(), "projection_id": projection_id.fetchone()[0]}


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
        INSERT INTO media (media_path, original_path, original_id, media_type, file_id, sub_type, size, metadata, library_id, hash, parent_id, start_ts, end_ts, start_frame, end_frame, frame_rate)
        VALUES (:media_path, :original_path, :original_id, :media_type, :file_id, :sub_type, :size, :metadata, :library_id, :hash, :parent_id, :start_ts, :end_ts, :start_frame, :end_frame, :frame_rate)
        RETURNING media_id
    """)
    media_id = DataAccessObject().execute_query(query, media_data)
    return {**media_data, "media_id": media_id.fetchone()[0]}


def read_media_by_id(media_id: int) -> dict:
    query = text("SELECT * FROM media WHERE media_id = :media_id")
    return DataAccessObject().fetch_one(query, {"media_id": media_id})


def read_media_by_library_id(library_id: int, media_type: str = None, sub_type: str = None) -> dict:
    query = "SELECT * FROM media WHERE library_id = :library_id"
    if media_type:
        query = query + " AND media_type = :media_type"
    if sub_type:
        query = query + " AND sub_type = :sub_type"

    query = text(query)
    return DataAccessObject().fetch_all(query, {"library_id": library_id, "media_type": media_type, "sub_type": sub_type})


def read_media_by_source_file_id(file_id: str) -> dict:
    query = text("SELECT * FROM media WHERE file_id = :file_id")
    return DataAccessObject().fetch_one(query, {"file_id": file_id})


def read_media() -> dict:
    query = text("SELECT * FROM media")
    return DataAccessObject().fetch_all(query)


def read_media_by_metadata(key: str, value: str) -> dict:
    query = text("SELECT * FROM media WHERE metadata->>:key = :value")
    return DataAccessObject().fetch_all(query, {"key": key, "value": value})


def read_media_by_feature_data(key: str, value: str) -> dict:

    # query feature table and join with media table
    query = text("""
        SELECT * FROM media
        JOIN feature ON media.media_id = feature.media_id
        WHERE feature.data->>:key LIKE :value
    """)
    # WHERE data->>'ner' LIKE '%{person}%';
    return DataAccessObject().fetch_all(query, {"key": key, "value": f"%{value}%"})


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


def create_feature(feature: Feature):
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


def read_feature_by_id(feature_id: int):
    query = text("SELECT * FROM feature WHERE feature_id = :feature_id")
    result = DataAccessObject().fetch_one(query, {"feature_id": feature_id})
    return result


def get_features():
    query = text("SELECT * FROM feature")
    result = DataAccessObject().fetch_all(query)
    return result


def update_feature(feature_id: int, feature: Feature):
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
    return result


def delete_feature(feature_id: int):
    query = text("""
        DELETE FROM feature
        WHERE feature_id = :feature_id
        RETURNING *
    """)
    result = DataAccessObject().fetch_one(query, {"feature_id": feature_id})
    return result


def get_nearest_neighbors(media_id: int, feature_type: str, model_name: str, version: str, k: int = 10):
    _query = """
        WITH target_embedding AS (
        SELECT
            media_id,
            CASE
                WHEN embedding_size = 1024 THEN embedding_1024
                WHEN embedding_size = 2048 THEN embedding_2048
                ELSE NULL
            END AS embedding_vector
        FROM 
            feature
        WHERE 
            media_id = :media_id AND feature_type = :feature_type AND model_name = :model_name 
        )

        SELECT
        f.media_id,
        (target.embedding_vector <-> 
            CASE
            WHEN f.embedding_size = 1024 THEN f.embedding_1024
            WHEN f.embedding_size = 2048 THEN f.embedding_2048
            ELSE NULL
            END
        ) AS distance
        FROM
        feature f,
        target_embedding target
        WHERE
        f.media_id != target.media_id
        ORDER BY
        distance ASC
        LIMIT :k;
    """
    query = text(_query)
    result = DataAccessObject().fetch_all(
        query, {"media_id": media_id, "k": k, "feature_type": feature_type, "model_name": model_name, "version": version})
    return result


def create_map_projection_feature(map_projection_feature: MapProjectionFeatureCreate):
    query = text("""
        INSERT INTO map_projection_feature (projection_id, feature_id, media_id, atlas_order, coordinates, index_in_atlas)
        VALUES (:projection_id, :feature_id, :media_id, :atlas_order, ST_MakePoint(:x, :y, :z), :index_in_atlas)
        RETURNING map_projection_feature_id, projection_id, feature_id, media_id, atlas_order, ST_AsText(coordinates) as coordinates, index_in_atlas
    """)
    params = map_projection_feature.dict()
    try:
        params["x"], params["y"], params["z"] = map_projection_feature.coordinates
    except ValueError:
        params["x"], params["y"] = map_projection_feature.coordinates
        params["z"] = 0
    result = DataAccessObject().execute_query(query, params)
    return result


def read_map_projection_features():
    query = """
    SELECT map_projection_feature_id, projection_id, feature_id, media_id, atlas_order, ST_AsText(coordinates) as coordinates
    FROM map_projection_feature
    """
    result = DataAccessObject().fetch_all(query)
    return result


def get_projection_coordinates(projection_id: int):
    query = text("""
        SELECT ST_X(map_projection_feature.coordinates) as x, ST_Y(map_projection_feature.coordinates) as y,
            media.media_path
        FROM map_projection_feature
        LEFT JOIN media ON media.media_id = map_projection_feature.media_id
        WHERE map_projection_feature.projection_id = :projection_id

    """)
    result = DataAccessObject().fetch_all(
        query, {"projection_id": projection_id})
    return result


def get_projection_coordinates_by_atlas(projection_id: int, atlas_order: int):
    query = text("""
        SELECT ST_X(map_projection_feature.coordinates) as x, ST_Y(map_projection_feature.coordinates) as y,
            media.media_path
        FROM map_projection_feature
        LEFT JOIN media ON media.media_id = map_projection_feature.media_id
        WHERE map_projection_feature.projection_id = :projection_id
        AND map_projection_feature.atlas_order = :atlas_order
    """)
    result = DataAccessObject().fetch_all(
        query, {"projection_id": projection_id, "atlas_order": atlas_order})
    return result


def get_projection_atlases(projection_id: int):
    query = text("""
        SELECT atlas_id, atlas_path, atlas_size, tile_size, tile_count, rows, cols, tiles_per_atlas
        FROM atlas
        WHERE projection_id = :projection_id
    """)
    result = DataAccessObject().fetch_all(
        query, {"projection_id": projection_id})
    return result


def get_atlas(atlas_id: int):
    query = text("SELECT * FROM atlas WHERE atlas_id = :atlas_id")
    return DataAccessObject().fetch_one(query, {"atlas_id": atlas_id})


def get_atlases():
    query = text("SELECT * FROM atlas")
    return DataAccessObject().fetch_all(query)


def create_atlas(atlas: AtlasCreate):

    query = text("""
        INSERT INTO atlas (projection_id, atlas_order, atlas_path, atlas_size, tile_size, tile_count, rows, cols, tiles_per_atlas)
        VALUES (:projection_id, :atlas_order, :atlas_path, :atlas_size, :tile_size, :tile_count, :rows, :cols, :tiles_per_atlas)
        RETURNING atlas_id
        """)
    atlas_dict = atlas.dict()
    result = DataAccessObject().execute_query(query, atlas_dict)
    return {**atlas_dict, "atlas_id": result.fetchone()[0]}
