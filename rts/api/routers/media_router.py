from typing import List
from fastapi import FastAPI
from sqlalchemy.sql import text
from models import Media  # Assuming models.py contains your Pydantic models
import json

app = FastAPI()


@app.post("/media/", response_model=Media)
async def create_media(media: Media):
    media_data = media.dict()
    media_data['metadata'] = json.dumps(media_data['metadata'])

    query = text("""
        INSERT INTO media (media_path, original_path, media_type, sub_type, size, metadata, library_id, hash, parent_id, start_ts, end_ts, start_frame, end_frame, frame_rate)
        VALUES (:media_path, :original_path, :media_type, :sub_type, :size, :metadata, :library_id, :hash, :parent_id, :start_ts, :end_ts, :start_frame, :end_frame, :frame_rate)
    """)
    db.execute_query(query, media_data)
    return media_data


@app.get("/media/", response_model=List[Media])
async def read_medias():
    query = text("SELECT * FROM media")
    result = db.fetch_all(query)
    for row in result:
        row['metadata'] = json.loads(row['metadata'])
    return result


@app.get("/media/{media_id}", response_model=Media)
async def read_media(media_id: int):
    query = text("SELECT * FROM media WHERE media_id = :media_id")
    result = db.fetch_one(query, {"media_id": media_id})
    if result:
        result['metadata'] = json.loads(result['metadata'])
    return result


@app.put("/media/{media_id}", response_model=Media)
async def update_media(media_id: int, media: Media):
    media_data = media.dict()
    media_data['metadata'] = json.dumps(media_data['metadata'])

    query = text("""
        UPDATE media
        SET media_path=:media_path, original_path=:original_path, media_type=:media_type, sub_type=:sub_type, size=:size, metadata=:metadata, library_id=:library_id, hash=:hash, parent_id=:parent_id, start_ts=:start_ts, end_ts=:end_ts, start_frame=:start_frame, end_frame=:end_frame, frame_rate=:frame_rate
        WHERE media_id=:media_id
    """)
    db.execute_query(query, {**media_data, "media_id": media_id})
    return {**media_data, "media_id": media_id}


@app.delete("/media/{media_id}")
async def delete_media(media_id: int):
    query = text("DELETE FROM media WHERE media_id = :media_id")
    db.execute_query(query, {"media_id": media_id})
    return {"status": "Media deleted"}
