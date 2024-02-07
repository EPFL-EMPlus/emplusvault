import os
import sys
import time
import logging
import orjson
import pickle
import contextlib
import pandas as pd
import cv2

from typing import Dict, List, Optional, Tuple, Callable, Any, Generator
from functools import wraps
from pathlib import Path
from queue import Queue
from threading import Thread
import tempfile


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
            js = orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)
            fp.write(js)
            return True
    except (ValueError, orjson.JSONEncodeError) as e:
        import traceback
        LOG.error(e)
        LOG.error(traceback.format_exc())
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


def dataframe_to_hdf5(outdir: str, name: str, df: pd.DataFrame) -> bool:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    try:
        df.to_hdf(os.path.join(outdir, f'{name}.hdf5'), key='df', mode='w')
        return True
    except Exception as e:
        return False
    

def dataframe_from_hdf5(input_dir: str, name: str, key: str = 'df', silent: bool = False) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_hdf(os.path.join(input_dir, f'{name}.hdf5'), key=key)
        return df
    except Exception as e:
        if not silent:
            LOG.error(e)
        return None


@contextlib.contextmanager
def temporary_filename(suffix=None):
  # https://stackoverflow.com/questions/3924117/how-to-use-tempfile-namedtemporaryfile-in-python
  """Context that introduces a temporary file.

  Creates a temporary file, yields its name, and upon context exit, deletes it.
  (In contrast, tempfile.NamedTemporaryFile() provides a 'file' object and
  deletes the file as soon as that file object is closed, so the temporary file
  cannot be safely re-opened by another library or process.)

  Args:
    suffix: desired filename extension (e.g. '.mp4').

  Yields:
    The name of the temporary file.
  """
  import tempfile
  try:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_name = f.name
    f.close()
    yield tmp_name
  finally:
    os.unlink(tmp_name)


def read_xml_file(file_path: str):
    with open(file_path, 'r') as file:
        xml_string = file.read()
    return xml_string


def read_mpd_file(media_folder: str):
    p = Path(media_folder)
    media_id = p.name
    mpd_file = p / Path(media_id + '.mpd')
    return read_xml_file(str(mpd_file))


def remove_path_prefix(text: str, prefix: str) -> str:
    return text[text.startswith(prefix) and len(prefix):]


class FileVideoStream:
    def __init__(self, video_bytes, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        self.tmp_file.write(video_bytes)
        self.tmp_file.close()
        self.stream = cv2.VideoCapture(self.tmp_file.name)

        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
    
    def __del__(self):
        # Clean up the temporary file when the object is destroyed
        os.remove(self.tmp_file.name)

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True
                    
                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()