from fastapi import APIRouter
from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from rts.db.dao import DataAccessObject
from rts.api.models import LibraryBase, LibraryCreate, Library
from rts.utils import get_logger

LOG = get_logger()

library_router = APIRouter() 

# TODO: separate the functions so that they can be called not only over the API but also from code
@library_router.get("/libraries/{library_id}", response_model=Library)
async def read_library(library_id: int):

    # query using asyncpg
    query = "SELECT * FROM library WHERE library_id=$1 or library_name=$2"
    library_name = "rts"
    library = await DataAccessObject().fetch_one(query, (library_id, library_name))
    if library is None:
        raise HTTPException(status_code=404, detail="Library not found")
    return library


@library_router.post("/libraries/", response_model=Library)
async def create_library(library: LibraryBase):
    query = """
        INSERT INTO library (library_name, version, data)
        VALUES (:library_name, :version, :data)
        RETURNING library_id
    """
    LOG.info(f"Inserting new row {library.library_name}")
    library_id = await DataAccessObject().execute_query(query, library.dict())
    return {**library.dict(), "library_id": library_id}
