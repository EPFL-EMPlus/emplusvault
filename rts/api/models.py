from pydantic import BaseModel, Field
import datetime

class LibraryBase(BaseModel):
    library_name: str
    version: str

class LibraryCreate(LibraryBase):
    pass

class Library(LibraryBase):
    library_id: int
    created_at: datetime.date

