from fastapi import APIRouter
from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from rts.db.dao import DataAccessObject
from rts.api.models import LibraryBase, LibraryCreate, Library
from rts.utils import get_logger
import sqlalchemy
import databases
import json

LOG = get_logger()

library_router = APIRouter() 

# TODO: as we are not using an ORM, do we need to be careful about SQL injection?
# TODO: separate the functions so that they can be called not only over the API but also from code
@library_router.get("/libraries/{library_id}")
async def read_library(library_id: int):

    # put query as sqlalchemy statement to resolve it to a string

    # query using asyncpg
    query = "SELECT * FROM library WHERE library_id=%s or library_name=%s"
    library_name = "rts"
    library = DataAccessObject().fetch_one(query, (library_id, library_name))

    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library


@library_router.post("/libraries/")
def create_library(library: LibraryBase):
    
    query = """
        INSERT INTO library (library_name, version, data)
        VALUES (%s, %s, %s)
        RETURNING library_id
    """
    vals = library.dict()
    library_id = DataAccessObject().execute_query(query, (vals['library_name'], vals['version'], json.dumps(vals['data'])))
    return {**library.dict(), "library_id": library_id.fetchone()[0]}
