from fastapi import APIRouter
from fastapi import (APIRouter, Depends, Request, Response, HTTPException)
from rts.api.dao import DataAccessObject
from models import LibraryBase, LibraryCreate, Library


library_router = APIRouter()

@library_router.get("/libraries/{library_id}", response_model=Library)
async def read_library(library_id: int):
    query = "SELECT * FROM library WHERE library_id=:id"
    library = await DataAccessObject().fetch_one(query, {"id": library_id})
    if library is None:
        raise HTTPException(status_code=404, detail="Library not found")
    return library
