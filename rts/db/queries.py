from rts.db.dao import DataAccessObject
from typing import Optional
from rts.api.models import LibraryBase
import json


def get_library_id_from_name(library_name: str) -> Optional[int]:
    query = """
        SELECT library_id FROM library WHERE library_name=%s
    """
    library_id = DataAccessObject().fetch_one(query, (library_name,))
    return library_id['library_id'] if library_id else None


def create_new_library(library: LibraryBase):
    query = """
        INSERT INTO library (library_name, version, data)
        VALUES (%s, %s, %s)
        RETURNING library_id
    """
    vals = library.dict()
    library_id = DataAccessObject().execute_query(
        query, (vals['library_name'], vals['version'], json.dumps(vals['data'])))
    return {**library.dict(), "library_id": library_id.fetchone()[0]}
