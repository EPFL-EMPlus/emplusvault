from rts.db.dao import DataAccessObject
from typing import Optional


def get_library_id_from_name(library_name: str) -> Optional[int]:
    query = """
        SELECT library_id FROM library WHERE library_name=%s
    """
    library_id = DataAccessObject().fetch_one(query, (library_name,))
    return library_id['library_id'] if library_id else None
