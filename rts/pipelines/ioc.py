import pandas as pd
import rts.utils

from sqlalchemy.exc import IntegrityError
from datetime import datetime

from rts.pipelines.base import Pipeline, get_hash
from rts.io.media import get_frame_number
from rts.api.models import Media
from rts.db.queries import create_or_update_media


LOG = rts.utils.get_logger()


IOC_ROOT = "/media/data/ioc/"
IOC_DATA = IOC_ROOT + 'data'
IOC_VIDEOS = IOC_ROOT + 'videos'


class PipelineIOC(Pipeline):
    library_name: str = 'ioc'

    def ingest(self, df: pd.DataFrame, min_duration: float = 0, max_duration: float = float('inf')) -> bool:
        """
        Ingest the IOC metadata and video files.

        Parameters:
            df (pd.DataFrame): IOC metadata dataframe which is expected to have the following columns:
                - guid (str): unique identifier for the video
                - seq_id (str): sequence identifier for the clips
                - start (str): start time of the clip in the format HH:MM:SS:ff
                - end (str): end time of the clip in the format HH:MM:SS:ff
                - path (str): path to the video file
                - sport (str): sport
                - description (str): description of the video
                - event (str): event
                - category (str): category of the event
                - round (str): round of the event
        """
        for i, group in self.tqdm(df.groupby('guid')):
            self.ingest_single_video(group, min_duration, max_duration)

        return True

    def ingest_single_video(self, df: pd.DataFrame, 
                            min_duration: float = 0, 
                            max_duration: float = float('inf')) -> bool:
        """ Ingest all clips from a single IOC video file. """
        df = self.preprocess(df)
    
        for i, row in self.tqdm(df.iterrows(), leave=False, total=len(df)):
            seq_dur = row.end_ts - row.start_ts
            if seq_dur < min_duration:
                LOG.info(f'Skipping clip {row.seq_id} because it is shorter than {min_duration} seconds')
                continue

            if seq_dur > max_duration:
                LOG.info(f'Skipping clip {row.seq_id} because it is longer than {max_duration} seconds')
                continue

            original_path = row.path
            media_path = f"videos/{row.guid}/{row.seq_id}.mp4"

            media_info = self.trim_upload_media(original_path, media_path, row.start_ts, row.end_ts)
            if not media_info:
                LOG.error(f'Failed to trim and upload clip {row.seq_id}')
                return False

            metadata = {
                'sport': row.sport,
                'description': row.description,
                'event': row.event,
                'category': row.category,
                'round': row['round']
            }
            metadata['media_info'] = media_info

            clip = Media(**{
                'media_id': row.seq_id,
                'original_path': original_path,
                'original_id': row.guid,
                'media_path': media_path, 
                'media_type': "video",
                'sub_type': "clip", 
                'size': media_info['filesize'],
                'metadata': metadata,
                'library_id': self.library_id, 
                'hash': get_hash(media_path), 
                'parent_id': -1,
                'start_ts': row.start_ts, 
                'end_ts': row.end_ts, 
                'start_frame': get_frame_number(row.start_ts, media_info['video']['framerate']),
                'end_frame': get_frame_number(row.end_ts, media_info['video']['framerate']), 
                'frame_rate': media_info['video']['framerate'], 
            })

            try:
                create_or_update_media(clip)
            except IntegrityError as e:
                if "duplicate key value violates unique constraint" in str(e):
                    LOG.info(f'UniqueViolation: Duplicate media_id {clip.media_id}')
                else:
                    raise e
        return True

    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the actual starting point of the video. Annotations start a lot of time in the middle of the day, but we want to count from 0. 
        # e.g. they start for example at 12:01:02 and the next sequence is at 12:01:10, which we want to translate to 8s into the video
        df = df.copy()
        df.loc[:, 'start_ts'] = df.start.apply(lambda x: (datetime.strptime(x, '%H:%M:%S.%f') - datetime(1900, 1, 1)).total_seconds())
        df.loc[:, 'end_ts'] = df.end.apply(lambda x: (datetime.strptime(x, '%H:%M:%S.%f') - datetime(1900, 1, 1)).total_seconds())
        df.loc[:, 'end_ts'] = df.end_ts - df.start_ts.min()
        df.loc[:, 'start_ts'] = df.start_ts - df.start_ts.min()
        return df
    