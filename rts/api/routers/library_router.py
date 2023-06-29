from fastapi import (APIRouter, Depends, HTTPException)
from rts.db.dao import DataAccessObject
from rts.db.queries import create_library
from rts.api.models import LibraryBase
from rts.utils import get_logger
from rts.api.routers.auth_router import get_current_active_user
from supabase import Client
import json

LOG = get_logger()

library_router = APIRouter()

# TODO: as we are not using an ORM, do we need to be careful about SQL injection?
# TODO: separate the functions so that they can be called not only over the API but also from code


@library_router.get("/libraries/{library_id}")
async def read_library(library_id: int, supabase: Client = Depends(get_current_active_user)):

    # put query as sqlalchemy statement to resolve it to a string

    # query using asyncpg
    query = "SELECT * FROM library WHERE library_id=%s or library_name=%s"
    library_name = "rts"
    library = DataAccessObject().fetch_one(query, (library_id, library_name))

    if not library:
        raise HTTPException(status_code=404, detail="Library not found")
    return library


@library_router.get("/libraries/")
async def read_libraries(supabase: Client = Depends(get_current_active_user)):
    query = "SELECT * FROM library"
    libraries = DataAccessObject().fetch_all(query)

    if not libraries:
        raise HTTPException(status_code=404, detail="Libraries not found")
    return libraries


@library_router.get("/libraries/{library_id}/projections")
async def read_library_projections(library_id: int, supabase: Client = Depends(get_current_active_user)):
    query = "SELECT * FROM projection WHERE library_id=%s"
    projections = DataAccessObject().fetch_all(query, (library_id,))

    if not projections:
        raise HTTPException(status_code=404, detail="Projections not found")
    return projections


@library_router.post("/libraries/")
async def create(library: LibraryBase, supabase: Client = Depends(get_current_active_user)):
    return create_library(library)
