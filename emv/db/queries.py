import json
import emv.utils

from typing import Optional, Any, Dict
from sqlalchemy.sql import text
from sqlalchemy.exc import IntegrityError, ProgrammingError
from emv.db.dao import DataAccessObject
from emv.api.models import LibraryBase, Projection, Media, Feature, MapProjectionFeatureCreate, Atlas, AccessLog
from emv.api.routers.auth_router import get_password_hash, UserInDB as User
from fastapi import HTTPException

LOG = emv.utils.get_logger()


def get_library_id_from_name(library_name: str) -> Optional[int]:
    query = """
        SELECT library_id FROM library WHERE library_name=:library_name
    """
    library_id = DataAccessObject().fetch_one(text(query), {"library_name": library_name,})
    return library_id['library_id'] if library_id else None


def get_libraries() -> list:
    query = """
        SELECT * FROM library
    """
    return DataAccessObject().fetch_all(query)


# def remove_library(library_id: int) -> None:
#     # Reimplement if needed, removing a library is very involved
#     query = """
#         DELETE FROM library WHERE library_id=:library_id
#     """
#     DataAccessObject().execute_query(text(query), {"library_id": library_id})


def get_library_from_name(library_name: str) -> Optional[Dict]:
    query = """
        SELECT * FROM library WHERE library_name=:library_name
    """
    library = DataAccessObject().fetch_one(text(query), {"library_name": library_name})
    return library if library else None


def create_library(library: LibraryBase) -> dict:
    query = """
        INSERT INTO library (library_name, prefix_path, version, data)
        VALUES (:library_name, :prefix_path, :version, :data)
        RETURNING library_id
    """
    vals = library.dict()
    vals['data'] = json.dumps(vals['data'])
    library_id = DataAccessObject().execute_query(query, vals)
    return {**library.dict(), "library_id": library_id.fetchone()[0]}


def update_library_prefix_path(library_id: int, prefix_path: str) -> None:
    query = """
        UPDATE library SET prefix_path=:prefix_path WHERE library_id=:library_id
    """
    DataAccessObject().execute_query(query, {"library_id": library_id, "prefix_path": prefix_path})


def create_projection(projection: Projection) -> dict:
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


def get_projection_by_id(projection_id: int) -> Optional[dict]:
    query = text(
        "SELECT * FROM projection WHERE projection_id = :projection_id")
    return DataAccessObject().fetch_one(
        query, {"projection_id": projection_id})


def get_all_projections() -> list:
    query = text("SELECT * FROM projection")
    return DataAccessObject().fetch_all(query)


def update_projection(projection_id: int, projection: Projection) -> None:
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


def delete_projection(projection_id: int) -> None:
    query = text("DELETE FROM projection WHERE projection_id = :projection_id")
    DataAccessObject().execute_query(query, {"projection_id": projection_id})


def create_media(media: Media) -> dict:
    media_data = media.dict()
    media_data['metadata'] = json.dumps(
        media_data['metadata'], check_circular=False) if media_data['metadata'] else None
    media_data['media_info'] = json.dumps(
        media_data['media_info'], check_circular=False)

    query = text("""
        INSERT INTO media (media_id, media_path, original_path, original_id, media_type, media_info, sub_type, size, metadata, library_id, hash, parent_id, start_ts, end_ts, start_frame, end_frame, frame_rate)
        VALUES (:media_id, :media_path, :original_path, :original_id, :media_type, :media_info, :sub_type, :size, :metadata, :library_id, :hash, :parent_id, :start_ts, :end_ts, :start_frame, :end_frame, :frame_rate)
        RETURNING media_id
    """)
    media_id = DataAccessObject().execute_query(query, media_data)
    return {**media_data, "media_id": media_id.fetchone()[0]}


def create_or_update_media(media: Media) -> dict:
    media_data = media.dict()
    media_data['metadata'] = json.dumps(
        media_data['metadata'], check_circular=False) if media_data['metadata'] else None
    media_data['media_info'] = json.dumps(
        media_data['media_info'], check_circular=False)

    query = text("""
        INSERT INTO media (media_id, media_path, original_path, original_id, media_type, media_info, sub_type, size, metadata, library_id, hash, parent_id, start_ts, end_ts, start_frame, end_frame, frame_rate)
        VALUES (:media_id, :media_path, :original_path, :original_id, :media_type, :media_info, :sub_type, :size, :metadata, :library_id, :hash, :parent_id, :start_ts, :end_ts, :start_frame, :end_frame, :frame_rate)
        ON CONFLICT (media_id) DO UPDATE SET
        media_path = :media_path,
        original_path = :original_path,
        original_id = :original_id,
        media_type = :media_type,
        media_info = :media_info,
        sub_type = :sub_type,
        size = :size,
        metadata = :metadata,
        library_id = :library_id,
        hash = :hash,
        parent_id = :parent_id,
        start_ts = :start_ts,
        end_ts = :end_ts,
        start_frame = :start_frame,
        end_frame = :end_frame,
        frame_rate = :frame_rate
        RETURNING media_id
    """)
    media_id = DataAccessObject().execute_query(query, media_data)
    return {**media_data, "media_id": media_id.fetchone()[0]}


def get_media_by_id(media_id: str) -> dict:
    query = text("SELECT * FROM media WHERE media_id = :media_id")
    return DataAccessObject().fetch_one(query, {"media_id": str(media_id)})


def get_media_for_streaming(media_id: str) -> dict:
    query = text("SELECT media.media_path, media.size, library.library_name FROM media LEFT JOIN library ON media.library_id = library.library_id WHERE media_id = :media_id")
    return DataAccessObject().fetch_one(query, {"media_id": str(media_id)})


def check_media_exists(media_id: str) -> bool:
    query = text(
        "SELECT EXISTS(SELECT 1 FROM media WHERE media_id = :media_id)")
    result = DataAccessObject().fetch_one(query, {"media_id": media_id})
    return result['exists']


def get_all_media_by_library_id(library_id: int, last_seen_date: str = None, last_seen_media_id: str = None, page_size: int = 20, media_type: str = None, sub_type: str = None) -> dict:
    query = "SELECT * FROM media WHERE library_id = :library_id"

    if media_type:
        query = query + " AND media_type = :media_type"
    if sub_type:
        query = query + " AND sub_type = :sub_type"

    # Add conditions for the last_seen values
    if last_seen_date and last_seen_media_id:
        query += " AND (created_at, media_id) > (:last_seen_date, :last_seen_media_id)"

    # Order by both created_at and media_id
    query += " ORDER BY created_at, media_id LIMIT :page_size"

    params = {
        "library_id": int(library_id),
        "page_size": page_size
    }

    if media_type:
        params["media_type"] = media_type
    if sub_type:
        params["sub_type"] = sub_type
    if last_seen_date:
        params["last_seen_date"] = last_seen_date
    if last_seen_media_id:
        params["last_seen_media_id"] = int(
            last_seen_media_id)  # Ensure it's an integer

    return DataAccessObject().fetch_all(text(query), params)


def get_all_media() -> dict:
    query = text("SELECT * FROM media")
    return DataAccessObject().fetch_all(query)


def get_all_media_by_metadata(key: str, value: str) -> dict:
    query = text("SELECT * FROM media WHERE metadata->>:key = :value")
    return DataAccessObject().fetch_all(query, {"key": key, "value": value})


def get_media_by_feature_value(key: str, value: Any) -> dict:
    # query feature table and join with media table
    query = text("""
        SELECT * FROM media
        JOIN feature ON media.media_id = feature.media_id
        WHERE feature.data->>:key LIKE :value
    """)
    # WHERE data->>'ner' LIKE '%{person}%';
    return DataAccessObject().fetch_all(query, {"key": key, "value": f"%{value}%"})


def update_media(media_id: int, media: Media) -> dict:
    media_data = media.dict()
    media_data['metadata'] = json.dumps(media_data['metadata'])

    query = text("""
        UPDATE media
        SET media_path=:media_path, original_path=:original_path, media_type=:media_type, media_info=:media_info, sub_type=:sub_type, size=:size, metadata=:metadata, library_id=:library_id, hash=:hash, parent_id=:parent_id, start_ts=:start_ts, end_ts=:end_ts, start_frame=:start_frame, end_frame=:end_frame, frame_rate=:frame_rate
        WHERE media_id=:media_id
    """)
    DataAccessObject().execute_query(
        query, {**media_data, "media_id": media_id})
    return {**media_data, "media_id": media_id}


def delete_media(media_id: int) -> dict:
    query = text("DELETE FROM media WHERE media_id = :media_id")
    DataAccessObject().execute_query(query, {"media_id": media_id})
    return {"status": "Media deleted"}


def create_feature(feature: Feature) -> dict:
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


def get_feature_by_id(feature_id: int) -> dict:
    query = text("SELECT * FROM feature WHERE feature_id = :feature_id")
    result = DataAccessObject().fetch_one(query, {"feature_id": feature_id})
    return result


def get_feature_by_media_id(media_id: str, feature_type: str) -> dict:
    query = text(
        "SELECT feature_id FROM feature WHERE media_id = :media_id AND feature_type = :feature_type")
    result = DataAccessObject().fetch_one(
        query, {"media_id": media_id, "feature_type": feature_type})
    return result


def get_features_by_type(feature_type: str) -> dict:
    query = text(
        "SELECT * FROM feature WHERE feature_type = :feature_type")
    result = DataAccessObject().fetch_all(
        query, {"feature_type": feature_type})
    return result


def get_features_by_type_paginated(feature_type: str, page_size: int = 20, last_seen_feature_id: int = -1) -> dict:
    query = "SELECT * FROM feature WHERE feature_type = :feature_type"

    # Add conditions for the last_seen values
    if last_seen_feature_id:
        query += " AND feature_id > :last_seen_feature_id"

    # Order by both created_at and media_id
    query += " ORDER BY feature_id LIMIT :page_size"

    params = {
        "feature_type": feature_type,
        "page_size": page_size
    }

    if last_seen_feature_id:
        params["last_seen_feature_id"] = last_seen_feature_id

    return DataAccessObject().fetch_all(text(query), params)


def count_features_by_type(feature_type: str) -> dict:
    query = text(
        "SELECT COUNT(*) FROM feature WHERE feature_type = :feature_type")
    result = DataAccessObject().fetch_one(
        query, {"feature_type": feature_type})
    return result


def get_feature_data_by_media_id(media_id: str, feature_type: str) -> dict:
    query = text(
        "SELECT data FROM feature WHERE media_id = :media_id AND feature_type = :feature_type")
    result = DataAccessObject().fetch_one(
        query, {"media_id": media_id, "feature_type": feature_type})
    return result


def get_feature_by_media_id_and_type(media_id: str, feature_type: str) -> dict:
    query = text(
        "SELECT * FROM feature WHERE media_id = :media_id AND feature_type = :feature_type")
    result = DataAccessObject().fetch_one(
        query, {"media_id": media_id, "feature_type": feature_type})
    return result


def get_all_features() -> list:
    query = text("SELECT * FROM feature")
    result = DataAccessObject().fetch_all(query)
    return result


def update_feature(feature_id: int, feature: Feature) -> dict:
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
    result = DataAccessObject().execute_query(
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


def delete_feature_by_type(feature_type: str):
    query = text("""
        DELETE FROM feature
        WHERE feature_type = :feature_type
        RETURNING *
    """)
    result = DataAccessObject().fetch_one(
        query, {"feature_type": feature_type})
    return result


def get_nearest_neighbors(media_id: int, feature_type: str, model_name: str, version: str, k: int = 10) -> dict:
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


def get_atlas_by_projection_id_and_order(projection_id, atlas_order):
    query = text("""
        SELECT * FROM atlas
        WHERE projection_id = :projection_id AND atlas_order = :atlas_order
    """)
    result = DataAccessObject().fetch_all(
        query, {"projection_id": projection_id, "atlas_order": atlas_order})
    return result


def create_atlas(atlas: Atlas):

    query = text("""
        INSERT INTO atlas (projection_id, atlas_order, atlas_path, atlas_size, tile_size, tile_count, rows, cols, tiles_per_atlas)
        VALUES (:projection_id, :atlas_order, :atlas_path, :atlas_size, :tile_size, :tile_count, :rows, :cols, :tiles_per_atlas)
        RETURNING atlas_id
        """)
    atlas_dict = atlas.dict()
    result = DataAccessObject().execute_query(query, atlas_dict)
    return {**atlas_dict, "atlas_id": result.fetchone()[0]}


def create_new_user(user: User) -> dict:
    query = text(
        """
        INSERT INTO 
            users (username, full_name, email, password) 
            VALUES (:username, :full_name, :email, :password)
            RETURNING user_id
    """)

    user_dict = user.dict()
    user_dict["password"] = get_password_hash(
        user_dict["password"])

    try:
        result = DataAccessObject().execute_query(query, user_dict)
    except IntegrityError as e:
        if 'unique constraint' in str(e.orig).lower():
            LOG.error(
                'Username or email already exists. Please choose a different username.')
        else:
            LOG.error('An error occurred while inserting data into the database.')
        return None
    return {**user_dict, "user_id": result.fetchone()[0]}


def allow_user_to_access_library(user_id: int, library_id: int) -> bool:
    query = text(
        """
        INSERT INTO 
            user_library_access (user_id, library_id) 
            VALUES (:user_id, :library_id)
    """)
    DataAccessObject().execute_query(
        query, {"user_id": user_id, "library_id": library_id})
    return True


def check_access(current_user: User, media_id: str) -> bool:
    """ Checks if the current user has access to the media_id and logs the access. Throws an exception if the user does not have access.
    """
    query = text("""
        SELECT EXISTS(
            SELECT 1 FROM user_library_access 
            JOIN media ON media.library_id = user_library_access.library_id
            WHERE user_id = :user_id AND media_id = :media_id
        )        
    """)
    result = DataAccessObject().fetch_one(
        query, {"user_id": current_user.user_id, "media_id": media_id})

    if not result['exists']:
        raise HTTPException(status_code=401, detail="Not allowed")

    query = text(
        """
        INSERT INTO 
            access_logs (user_id, media_id) 
            VALUES (:user_id, :media_id)
    """)
    DataAccessObject().execute_query(
        query, {"user_id": current_user.user_id, "media_id": media_id})
    return True

def get_feature_wout_embedding_1024(feature_type: str, limit: int = 100) -> dict:
    query = text("""
        SELECT * FROM feature WHERE feature_type=:feature_type AND embedding_1024 IS NULL LIMIT :limit
    """)
    result = DataAccessObject().fetch_all(query, {"feature_type": feature_type, "limit": limit})
    return result
