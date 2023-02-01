import os
import sys
import time
import logging
import orjson
import pickle
from typing import Dict, List, Optional, Tuple, Callable, Any, Generator
from functools import wraps
from pathlib import Path


LOG = logging.getLogger('RTS')
LOG.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_logger() -> logging.Logger:
    return LOG


def timeit(method) -> Callable:
    @wraps(method)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        LOG.debug(f"{method.__name__} => {(end_time-start_time)*1000} ms")

        return result

    return wrapper


def human_readable_size(size: int , decimal_places: int = 2) -> str:
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def human_readable_bitrate(size: int , decimal_places: int = 2) -> str:
    for unit in ['B', 'K', 'M', 'G', 'T', 'P']:
        if size < 1000.0 or unit == 'P':
            break
        size /= 1000.0
    return f"{size:.{decimal_places}f}{unit}"


def read_jsonlines_file(filepath: str) -> List[Dict]:
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data.append(orjson.loads(line))
    except (FileNotFoundError, ValueError, orjson.JSONDecodeError) as e:
        LOG.error(e)
        return None
    return data


def obj_to_json(obj: Any, outpath: str) -> bool:
    try:
        with open(str(outpath), 'wb') as fp:
            js = orjson.dumps(obj)
            fp.write(js)
            return True
    except (ValueError, orjson.JSONEncodeError) as e:
        LOG.error(e)
        return False


def obj_from_json(path: str) -> Optional[Any]:
    try:
        with open(str(path), 'rb') as fp:
            js = orjson.loads(fp.read())
            return js
    except (FileNotFoundError, ValueError, orjson.JSONDecodeError) as e:
        LOG.error(e)
        return None


def dict_to_pickle(obj: Dict, outpath: str) -> bool:
    try:
        with open(str(outpath), 'wb') as fp:
            pickle.dump(obj, fp)
    except Exception as e:
        LOG.error(e)
        return False
    return True


def dict_from_pickle(path: str) -> Optional[Dict]:
    try:
        with open(str(path), 'rb') as fp:
            obj = pickle.load(fp)
    except Exception as e:
        LOG.error(e)
        return None
    return obj


def get_media_id(media_folder: str) -> str:
    media_id = Path(media_folder).name
    return media_id


def recursive_glob(path, file_extension='.json', match_name: Optional[str] = None) -> Generator[str, None, None]:
   for (dirpath, _, filenames) in os.walk(path):
      for filename in filenames:
         if filename.endswith(file_extension):
            if match_name:
               if filename == match_name:
                  yield os.path.join(dirpath, filename)
            else:
                yield os.path.join(dirpath, filename)
