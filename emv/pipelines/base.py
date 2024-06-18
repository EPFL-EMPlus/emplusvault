import hashlib
import pandas as pd

from abc import ABC, abstractmethod
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from emv.db.queries import get_library_from_name, get_library_id_from_name
from emv.utils import temporary_filename
from emv.io.media import trim, get_media_info
from emv.storage.storage import get_storage_client


def get_hash(media_path: str) -> str:
    return hashlib.md5(media_path.encode()).hexdigest()


class Pipeline(ABC):
    library_name: str = 'undefined'
    library: dict = {}
    store: object = None
    tqdm: object = None

    def __init__(self, library_name: str = None, run_notebook: bool = False):
        if library_name is None:
            library_name = self.library_name
        self.library = get_library_from_name(self.library_name)

        if not self.library:
            raise ValueError(
                f'Library {self.library_name} not found. Please initalize it first.')

        self.library_id = get_library_id_from_name(self.library_name)
        self.store = get_storage_client()
        self.tqdm = tqdm
        if run_notebook:
            self.tqdm = tqdm_notebook

    @abstractmethod
    def ingest(self, df: pd.DataFrame, force: bool = False, **kwargs) -> bool:
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    def trim_upload_media(self, source_media_path: str, outpath: str, start: float, end: float) -> dict:
        with temporary_filename('.mp4') as temp_video_path:
            ok, _ = trim(source_media_path, temp_video_path, start, end)
            if not ok:
                return None

            media_info = get_media_info(temp_video_path)
            media_info['path'] = outpath
            ok = self.store.upload(self.library_name, outpath, temp_video_path)
            if not ok:
                return None

            return media_info
