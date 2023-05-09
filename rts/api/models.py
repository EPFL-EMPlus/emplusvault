from pydantic import BaseModel, Field, Json
import datetime

class LibraryBase(BaseModel):
    library_name: str
    version: str
    data: Json

class LibraryCreate(LibraryBase):
    pass

class Library(LibraryBase):
    library_id: int
    created_at: datetime.date

