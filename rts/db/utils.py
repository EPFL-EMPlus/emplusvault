import os
import psycopg2
from typing import Optional, Dict
from rts.db.dao import DataAccessObject
from rts.db_settings import DB_NAME
from rts.utils import get_logger

LOG = get_logger()


def database_exists():
    return DataAccessObject().database_exists(DB_NAME)


def create_database(sql_file):
    # Read sql file and split it into individual statements
    with open(sql_file, "r") as f:
        statements = f.read().split(";")

    # LOG.info(f"Applying {len(statements)} statements from {sql_file}")
    # Execute each statement
    for statement in statements:
        # print(statement)
        if statement.strip() != "":
            DataAccessObject().execute_query(statement)
            # print("-- query executed --")


def reset_database():
    # TODO: We need to apply the migrations after this command has been run
    create_database("db/tables.sql")
    # statements = [
    #     "DROP TABLE IF EXISTS map_projection_feature;",
    #     "DROP TABLE IF EXISTS atlas;",
    #     "DROP TABLE IF EXISTS projection;",
    #     "DROP TABLE IF EXISTS feature;",
    #     "DROP TABLE IF EXISTS media;",
    #     "DROP TABLE IF EXISTS library;",
    # ]
    # for statement in statements:
    #     print(statement)
    #     if statement.strip() != "":
    #         DataAccessObject().execute_query(statement)


def write_media_object_db(
        media_path: str,
        original_path: str,
        library_id: int,
        parent_id: Optional[int] = None,
        start_ts: Optional[int] = -1,
        end_ts: Optional[int] = -1,
        start_frame: Optional[int] = -1,
        end_frame: Optional[int] = -1,
        frame_rate: Optional[int] = -1,
        update_data: Optional[Dict] = {},
        file_size: Optional[int] = -1,
        hash: Optional[str] = "",
        media_type: str = 'video',
        media_sub_type: str = "clip") -> str:
    _query = f"""
        INSERT INTO media (
            media_path, original_path, media_type, sub_type, 
            size, library_id, metadata, hash, 
            parent_id, start_ts, end_ts, start_frame, 
            end_frame, frame_rate)
        VALUES ('{media_path}', '{original_path}', '{media_type}', '{media_sub_type}', 
            {file_size}, {library_id}, 
            '{update_data}', '{hash}', 
            {parent_id}, {start_ts}, {end_ts}, {start_frame}, 
            {end_frame}, {frame_rate})
        ON CONFLICT (media_id) DO NOTHING;

    """
    execute_write_query(_query)
