import pandas as pd
import emv.utils

from sqlalchemy.exc import IntegrityError
from datetime import datetime
import base64

from emv.pipelines.base import Pipeline, get_hash
from emv.io.media import get_frame_number
from emv.api.models import Media
from emv.db.queries import create_or_update_media, check_media_exists
from emv.db.queries import get_feature_by_media_id_and_type, create_feature, update_feature
from emv.api.models import Feature

from emv.settings import IOC_ROOT_FOLDER
from emv.features.pose import process_video, PifPafModel
from emv.db.dao import DataAccessObject

LOG = emv.utils.get_logger()


IOC_DATA = IOC_ROOT_FOLDER + 'data'
IOC_VIDEOS = IOC_ROOT_FOLDER + 'videos'
MODEL = PifPafModel.fast

class PipelineIOC(Pipeline):
    library_name: str = 'ioc'

    def ingest(self, df: pd.DataFrame,
               force: bool = False,
               min_duration: float = 0,
               max_duration: float = float('inf')) -> bool:
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
        DataAccessObject().set_user_id(1)
        self.inner_progress_bar = None
        for i, group in self.tqdm(df.groupby('guid'), position=0, leave=True, desc='IOC videos'):
            self.ingest_single_video(group, force, min_duration, max_duration)

        return True

    def ingest_single_video(self, df: pd.DataFrame,
                            force: bool = False,
                            min_duration: float = 0,
                            max_duration: float = float('inf')
                            ) -> bool:
        """ Ingest all clips from a single IOC video file. """

        # The first sequence in the dataframe needs to be the full video for correct time stamp calculation
        try:
            assert len(df[df.seq_id == df.guid]) >= 1
        except AssertionError:
            LOG.error(
                f'No guid section found for the video with ID {df.guid.iloc[0]}. Skipping ingestion.')
            return False
        try:
            df = self.preprocess(df)
        except ValueError:
            return False
        LOG.info(f'Ingesting {len(df)} clips from {df.guid.iloc[0]}')

        # Setting up nested progress bar in an alternative way as the default leads to a bug with a lot of blank lines in jupyter notebooks
        if not self.inner_progress_bar:
            self.inner_progress_bar = self.tqdm(
                total=1, desc='Clips', position=1, leave=False)
        self.inner_progress_bar.reset()
        self.inner_progress_bar.total = len(df)
        self.inner_progress_bar.refresh()

        # for i, row in self.tqdm(df.iterrows(), leave=False, total=len(df), position=1, desc='Clips'):
        for i, row in df.iterrows():
            self.inner_progress_bar.update()

            media_id = f"{self.library_name}-{row.seq_id}"
            exists = check_media_exists(media_id)
            if exists and not force:
                continue

            seq_dur = row.end_ts - row.start_ts
            if seq_dur < min_duration:
                LOG.info(
                    f'Skipping clip {row.seq_id} of length {seq_dur} because it is shorter than {min_duration} seconds')
                continue

            if seq_dur > max_duration:
                LOG.info(
                    f'Skipping clip {row.seq_id} of length {seq_dur} because it is longer than {max_duration} seconds')
                continue

            original_path = row.path
            media_path = f"videos/{row.guid}/{row.seq_id}.mp4"

            LOG.info(f"Trimming and uploading clip {original_path} {media_path}")
            media_info = self.trim_upload_media(
                original_path, media_path, row.start_ts, row.end_ts)
            if not media_info:
                LOG.error(f'Failed to trim and upload clip {row.seq_id}')
                return False

            metadata = {
                'sport': row.sport,
                'description': str(row.description),
                'event': row.event,
                'category': row.category,
                'round': row['round'],
            }

            clip = Media(**{
                'media_id': media_id,
                'original_path': original_path,
                'original_id': row.guid,
                'media_path': media_path,
                'media_type': "video",
                'media_info': media_info,
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
                    LOG.info(
                        f'UniqueViolation: Duplicate media_id {clip.media_id}')
                else:
                    raise e
        return True

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the actual starting point of the video. Annotations start a lot of time in the middle of the day, but we want to count from 0.
        # e.g. they start for example at 12:01:02 and the next sequence is at 12:01:10, which we want to translate to 8s into the video
        df = df.copy()
        df.loc[:, 'start_ts'] = df.start.apply(lambda x: (datetime.strptime(
            x, '%H:%M:%S.%f') - datetime(1900, 1, 1)).total_seconds())
        df.loc[:, 'end_ts'] = df.end.apply(lambda x: (datetime.strptime(
            x, '%H:%M:%S.%f') - datetime(1900, 1, 1)).total_seconds())
        df.loc[:, 'end_ts'] = df.end_ts - df.start_ts.min()
        df.loc[:, 'start_ts'] = df.start_ts - df.start_ts.min()
        df = df.sort_values(by='start_ts')
        return df


    def process_poses(self, df: pd.DataFrame) -> bool:
        """ Process the poses of all clips in the dataframe. """
        # we need a user id for RLS to work even while batch processing
        DataAccessObject().set_user_id(1)
        for i, row in self.tqdm(df.iterrows(), leave=False, total=len(df), position=1, desc='Clips'):
            self.process_single_pose(row)
        return True
    
    def process_single_pose(self, row: pd.Series) -> bool:
        """ Process the poses of a single clip. """
        result = self.store.get_bytes("ioc", f"videos/{row.guid}/{row.seq_id}.mp4")

        if not result:
            # We won't have a result if the video is not in the storage
            return False
        
        r, images = process_video(
            MODEL, 
            result,
            skip_frame=10
        )
        image_references = []
        # Upload to s3    
        for i, img in enumerate(images):
            img_path = f"images/{row.guid}/{row.seq_id}/pose_frame_{r[i]['frame']}.jpg"
            self.store.upload(self.library_name, img_path, img)
            
            media_id = f"ioc-{row.seq_id}-pose-{r[i]['frame']}"
            image_references.append(media_id)
            r[i]['media_id'] = media_id
            media_dict = {
                'media_id': media_id,
                'original_path': img_path,
                'original_id': row.guid,
                'media_path': img_path,
                'media_type': "image",
                'media_info': {'type': 'image/jpeg'},
                'sub_type': "screenshot/frame",
                'size': 0,
                'metadata': {'pose': 'openpifpaf'},
                'library_id': self.library_id if self.library_id else 2,
                'hash': get_hash(img_path),
                'parent_id': "ioc-" + row.seq_id,
                'start_ts': 0,
                'end_ts': 0,
                'start_frame': r[i]['frame'] if r[i]['frame'] else 0,
                'end_frame': r[i]['frame'] if r[i]['frame'] else 0,
                'frame_rate': 1,
            }
            LOG.debug(f'Creating media {media_id}')
            LOG.debug(media_dict)

            screenshot = Media(**media_dict)
            create_or_update_media(screenshot)

        clip_media_id = f"ioc-{row.seq_id}"
        feature_type = 'pose'

        feature = get_feature_by_media_id_and_type(clip_media_id, feature_type)

        new_feature = Feature(
            feature_type=feature_type,
            version=1,
            model_name='PifPafModel.fast',
            model_params={
                'PifPafModel': 'fast',
            },
            data={"frames": r, "images": image_references},
            media_id=clip_media_id,
        )
        try:
            if feature:
                update_feature(feature['feature_id'], new_feature)
            else:
                create_feature(new_feature)
        except IntegrityError as e:
            LOG.error(f'Failed to create feature for {clip_media_id}: {e}')
        return True