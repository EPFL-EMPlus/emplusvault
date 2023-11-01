import pandas as pd
import emv.utils

from sqlalchemy.exc import IntegrityError
from datetime import datetime

from emv.pipelines.base import Pipeline, get_hash
from emv.io.media import get_media_info, get_frame_number
from emv.api.models import Media
from emv.db.queries import create_media, create_or_update_media, check_media_exists

from emv.settings import MJF_ROOT_FOLDER

LOG = emv.utils.get_logger()


MJF_METADATA = MJF_ROOT_FOLDER + 'metadata'
MIF_VIDEOS = MJF_ROOT_FOLDER + 'videos'


class PipelineMJF(Pipeline):
    library_name: str = 'mjf'

    def ingest(self, df: pd.DataFrame, force: bool = False) -> bool:
        for i, row in self.tqdm(df.iterrows()):
            self.ingest_single_video(row, force)

        return True

    def ingest_single_video(self, row: pd.Series, force: bool = False) -> bool:

        # TODO upload media
        media_id = f"{self.library_name}-{row.song_id}"

        exists = check_media_exists(media_id)
        if exists and not force:
            return True

        local_path = self.library['prefix_path'] + row.path
        media_info = get_media_info(local_path)

        if not media_info:
            LOG.error(f"Skipping {row.path} (no media info)")
            return False

        metadata = {
            'title': row.title,
            'concert_id': row.concert_id,
            'concert_name': row.concert_name,
            'date': row.date,
            'location': row.location,
            'genre': row.genre,
            'top_genre': row.top_genre,
            'musicians': row.musicians,
            'instruments': row.instruments
        }

        clip = Media(**{
            'media_id': media_id,
            'original_path': row.path,
            'original_id': row.song_id,
            'media_path': row.path,
            'media_type': "video",
            'sub_type': "",
            'size': media_info['filesize'],
            'metadata': metadata,
            'media_info': media_info,
            'library_id': self.library['library_id'],
            'hash': get_hash(row.path),
            'parent_id': -1,
            'start_ts': 0,
            'end_ts': media_info['duration'],
            'start_frame': get_frame_number(0, media_info['video']['framerate']),
            'end_frame': get_frame_number(media_info['duration'], media_info['video']['framerate']),
            'frame_rate': media_info['video']['framerate'],
        })

        try:
            create_or_update_media(clip)
        except IntegrityError as e:
            if "duplicate key value violates unique constraint" in str(e):
                LOG.info(
                    f'UniqueViolation: Duplicate media_id {clip.media_id}')
            else:
                raise e

        LOG.debug(f"Created media {clip.media_id}, {clip.media_path}")
        return True

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
