from fastapi import (APIRouter, Depends, HTTPException)
from emv.db.dao import DataAccessObject
from emv.db.queries import create_library
from emv.api.models import LibraryBase
from emv.utils import get_logger
from sqlalchemy.sql import text
from emv.api.routers.auth_router import get_current_active_user, User
import json

LOG = get_logger()

library_router = APIRouter()

# TODO: as we are not using an ORM, do we need to be careful about SQL injection?
# TODO: separate the functions so that they can be called not only over the API but also from code


@library_router.get("/libraries/{library_id}")
async def read_library(library_id: int, current_user: User = Depends(get_current_active_user)):

    # put query as sqlalchemy statement to resolve it to a string

    # query using asyncpg
    query = text("SELECT * FROM library WHERE library_id = :library_id or library_name = :library_name")
    library_name = "rts"
    library = DataAccessObject().fetch_one(query, {"library_id": library_id, "library_name": library_name})

    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library


@library_router.get("/libraries/")
async def read_libraries(current_user: User = Depends(get_current_active_user)):
    query = text("SELECT * FROM library")
    libraries = DataAccessObject().fetch_all(query)

    if not libraries:
        raise HTTPException(status_code=404, detail="Libraries not found")
    return libraries


@library_router.get("/libraries/{library_id}/projections")
async def read_library_projections(library_id: int, current_user: User = Depends(get_current_active_user)):
    query = text("SELECT * FROM projection WHERE library_id = :library_id")
    projections = DataAccessObject().fetch_all(query, {"library_id": library_id})

    if not projections:
        raise HTTPException(status_code=404, detail="Projections not found")
    return projections


# @library_router.post("/libraries/")
# async def create(library: LibraryBase, current_user: User = Depends(get_current_active_user)):
#     return create_library(library)
