import os
import sys
import time
import logging
import orjson
from typing import Dict, List, Optional, Tuple, Callable
from functools import wraps


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
        LOG.info(f"{method.__name__} => {(end_time-start_time)*1000} ms")

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


def dict_to_json(obj: Dict, outpath: str) -> bool:
    try:
        with open(outpath, 'wb') as fp:
            js = orjson.dumps(obj)
            fp.write(js)
            return True
    except (ValueError, orjson.JSONEncodeError) as e:
        LOG.error(e)
        return False


def dict_from_json(path: str) -> Optional[Dict]:
    try:
        with open(path, 'rb') as fp:
            js = orjson.loads(fp.read())
            return js
    except (FileNotFoundError, ValueError, orjson.JSONDecodeError) as e:
        LOG.error(e)
        return None
