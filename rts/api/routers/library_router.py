from fastapi import APIRouter
from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from rts.db.dao import DataAccessObject
from rts.api.models import LibraryBase, LibraryCreate, Library


library_router = APIRouter()

@library_router.get("/libraries/{library_id}", response_model=Library)
async def read_library(library_id: int):
    query = "SELECT * FROM library WHERE library_id=:id"
    library = await DataAccessObject().fetch_one(query, {"id": library_id})
    if library is None:
        raise HTTPException(status_code=404, detail="Library not found")
    return library


@library_router.post("/libraries/", response_model=Library)
async def create_library(library: LibraryCreate):
    query = """
        INSERT INTO library (library_name, version, data)
        VALUES (:library_name, :version, :data)
        RETURNING library_id
    """
    library_id = await DataAccessObject().execute(query, library.dict())
    return {**library.dict(), "library_id": library_id}
