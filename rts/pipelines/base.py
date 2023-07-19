import hashlib
import pandas as pd

from abc import ABC, abstractmethod
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from rts.db.queries import get_library_id_from_name
from rts.utils import temporary_filename
from rts.io.media import trim, get_media_info
from rts.storage.storage import get_storage_client


def get_hash(media_path: str) -> str:
    return hashlib.md5(media_path.encode()).hexdigest()


class Pipeline(ABC):
    library_name: str = 'undefined'
    frame_rates: dict = {}
    library_id: int = -1

    def __init__(self, run_notebook: bool = False):
        self.library_id = get_library_id_from_name(self.library_name)
        self.store = get_storage_client()
        self.tqdm = tqdm
        if run_notebook:
            self.tqdm = tqdm_notebook

    @abstractmethod
    def ingest(self, df: pd.DataFrame, **kwargs) -> bool:
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