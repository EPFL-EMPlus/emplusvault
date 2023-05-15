import os
import psycopg2
from dotenv import load_dotenv
from typing import Optional, Dict

load_dotenv()

def create_conn() -> psycopg2.extensions.connection:
    from rts.api.server import app
    if app.testing:
        conn = psycopg2.connect(
            host=os.getenv("TEST_DB_HOST"),
            port=os.getenv("TEST_DB_PORT"),
            dbname=os.getenv("TEST_DB_NAME"),
            user=os.getenv("TEST_DB_USER"),
            password=os.getenv("TEST_DB_PASSWORD"),
        )
        return conn
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    return conn

def execute_read_query(query) -> list:
    conn = create_conn()
    with conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()
    return result

def execute_write_query(query) -> None:
    conn = create_conn()
    with conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()


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

