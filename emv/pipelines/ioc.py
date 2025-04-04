import pandas as pd
import numpy as np
import emv.utils

from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import text
from datetime import datetime
import base64
from io import BytesIO

from emv.pipelines.base import Pipeline, get_hash
from emv.io.media import get_frame_number
from emv.api.models import Media, Feature, Projection, MapProjectionFeatureCreate
from emv.db.queries import create_or_update_media, check_media_exists, create_projection, create_map_projection_feature
from emv.db.queries import get_feature_by_media_id_and_type, create_feature, update_feature

from emv.settings import IOC_ROOT_FOLDER
from emv.features.pose import process_video, PifPafModel, write_pose_to_binary_file, extract_pose_frames
from emv.features.pose import read_pose_from_binary_file, get_pose_feature_vector as build_pose_vector_with_flip
from emv.db.dao import DataAccessObject
from emv.storage.storage import get_storage_client


LOG = emv.utils.get_logger()


IOC_DATA = IOC_ROOT_FOLDER + 'data'
IOC_VIDEOS = IOC_ROOT_FOLDER + 'videos'
# MODEL = PifPafModel.fast
MODEL = "yolo11l-pose.pt"


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

        #  fill NaN in round with empty strings
        df['round'] = df['round'].fillna('')
        df['category'] = df['category'].fillna('')
        df['event'] = df['event'].fillna('')
        df['description'] = df['description'].fillna('')
        df['sport'] = df['sport'].fillna('')

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
                LOG.info(
                    f'Skipping clip {row.seq_id} because it already exists')
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

            LOG.info(
                f"Trimming and uploading clip {original_path} {media_path}")
            try:
                media_info = self.trim_upload_media(
                    original_path, media_path, row.start_ts, row.end_ts)
            except TypeError:
                LOG.error(
                    f'Failed to trim and upload clip {row.seq_id} {original_path}')
                return False

            if not media_info:
                LOG.error(f'Failed to trim and upload clip {row.seq_id}')
                return False

            metadata = {
                'sport': row.sport,
                'description': str(row.description),
                'event': row.event,
                'category': row.category,
                'round': row['round'] if row['round'] else "",
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
                'parent_id': "",
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
        result = self.store.get_bytes(
            "ioc", f"videos/{row.guid}/{row.seq_id}.mp4")

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

    def process_all_video_poses(self, df: pd.DataFrame, force: bool = False,) -> bool:
        """ Process every single frame of a clip and store the poses. """
        DataAccessObject().set_user_id(1)
        # filter df to be only for sequences
        df = df[df.seq_id != df.guid]
        for i, row in self.tqdm(df.iterrows(), leave=False, total=len(df), position=1, desc='Clips'):
            try:
                self.process_all_poses_clip(row, force)
            except ValueError:
                pass  # The error message is already logged in the function
        return True

    def process_all_poses_clip(self, row: pd.Series, force: bool = False,) -> bool:
        """ Process every single frame of a clip and store the poses. """
        LOG.info(f'Processing poses for {row.guid}/{row.seq_id}.mp4')
        result = self.store.get_bytes(
            "ioc", f"videos/{row.guid}/{row.seq_id}.mp4")
        LOG.info(f'Processing poses for {row.guid}/{row.seq_id}.mp4')
        if not result:
            raise ValueError(
                f'Failed to download video {row.guid}/{row.seq_id}.mp4')

        media_id = f"ioc-{row.seq_id}"
        bin_media_id = f"ioc-{row.seq_id}-binary"
        feature_type = 'pose-binary'

        # check if binary file already exists
        feature = get_feature_by_media_id_and_type(
            bin_media_id, feature_type)

        if feature and not force:
            LOG.info(f'Binary file for {bin_media_id} already exists')
            return

        LOG.info(f'Processing binary poses for {row.guid}/{row.seq_id}.mp4')

        r, images = process_video(
            MODEL,
            result
        )

        LOG.info(f'Extracting pose frames for {row.guid}/{row.seq_id}.mp4')

        video_resolution = r[0]['data']['width_height']
        pose_frames = extract_pose_frames(r)
        media_path = f"pose_binaries/{row.guid}/{row.seq_id}.bin"

        binary_file = Media(**{
            'media_id': bin_media_id,
            'original_path': media_path,
            'original_id': row.guid,
            'media_path': media_path,
            'media_type': "video",
            'media_info': {"resolution": video_resolution},
            'sub_type': "binary",
            'size': -1,
            'metadata': {},
            'library_id': self.library_id,
            'hash': get_hash(media_path),
            'parent_id': media_id,
            'start_ts': 0.0,
            'end_ts': 0.0,
            'start_frame': 0,
            'end_frame': -1,
            'frame_rate': -1,
        })

        try:
            media_id = create_or_update_media(binary_file)['media_id']
        except IntegrityError as e:
            if "duplicate key value violates unique constraint" in str(e):
                LOG.info(
                    f'UniqueViolation: Duplicate media_id {binary_file.media_id}')
            else:
                raise e

        feature = get_feature_by_media_id_and_type(
            "ioc-" + row.seq_id, feature_type)

        new_feature = Feature(
            feature_type=feature_type,
            version="1",
            model_name='binary',
            model_params={},
            data={},
            media_id=media_id,
        )

        try:
            if feature:
                feature_id = update_feature(feature['feature_id'], new_feature)[
                    'feature_id']
            else:
                feature_id = create_feature(new_feature)['feature_id']
        except IntegrityError as e:
            LOG.error(f'Failed to create feature for {bin_media_id}: {e}')

        no_frames = r[-1]['frame'] + 1
        write_pose_to_binary_file(int(feature_id), no_frames, 13,
                                  video_resolution, pose_frames, "ioc", media_path)

        return r, images

    def extract_poses_from_binary(self):
        """ Extracts poses from the binary files and creates a feature for each frame. """
        # Retrieve a list of all binary files
        query = text(
            "SELECT media_id, created_at, media_path, parent_id FROM media WHERE media_type = 'video' AND sub_type = 'binary'")
        df = pd.DataFrame(DataAccessObject().fetch_all(query), columns=[
                          "media_id", "created_at", "media_path", "parent_id"])

        #  iterate through the dataframe, 300 clips at a time
        for i in self.tqdm(range(0, len(df), 300), desc='Processing binary files', position=0, leave=True):
            batch = df.iloc[i:i + 300]
            self.extract_poses_from_binary_subset(batch)

    def extract_poses_from_binary_subset(self, df: pd.DataFrame) -> bool:
        """ Extracts poses from the binary files and creates a feature for each valid frame. """

        all_keypoints = []
        all_clip_indices = []
        all_frame_numbers = []
        all_person_ids = []

        for clip_idx, (_, row) in enumerate(df.iterrows()):
            binary_data = BytesIO()
            get_storage_client().download("ioc", row.media_path, binary_data)
            _, _, _, _, video_resolution, frames = read_pose_from_binary_file(
                binary_data)

            for pose in frames:
                frame_number = pose[0]  # Original frame number in video
                person_id = pose[1]
                keypoints = np.array(pose[2])  # shape (13, 2)

                if np.any(keypoints == 0):
                    continue  # Skip incomplete poses

                all_keypoints.append(keypoints)
                all_clip_indices.append(clip_idx)
                all_frame_numbers.append(frame_number)
                all_person_ids.append(person_id)

        # At this point:
        # - all_keypoints[i] is the i-th valid pose (np.array)
        # - all_clip_indices[i] is the index of the clip in df
        # - all_frame_numbers[i] is the frame number in the original video

        # Select diverse poses
        selected_indices = self.select_diverse_poses(
            poses=all_keypoints,
            build_embedding_fn=build_pose_vector_with_flip,
            frame_lengths=np.bincount(all_clip_indices).tolist(),
            threshold=200.0,
        )

        print("Selection done, creating features")

        for i in selected_indices:
            pose = all_keypoints[i].flatten().tolist()
            clip_idx = all_clip_indices[i]
            frame_number = all_frame_numbers[i]
            person_id = all_person_ids[i]
            media_id = df.iloc[clip_idx].parent_id

            # Reshape the keypoints from flat list [x1, y1, x2, y2, ...] to nested list [[x1, y1], [x2, y2], ...]
            keypoints_nested = []
            for j in range(0, len(pose), 2):
                if j + 1 < len(pose):
                    keypoints_nested.append([pose[j], pose[j+1]])

            feature = Feature(
                feature_type='pose-binary-extracted',
                version="1",
                model_name='YOLO.pose',
                model_params={'YOLO': 'pose'},
                data={
                    'keypoints': keypoints_nested,
                    'frame': frame_number,
                    'person_id': person_id,
                },
                media_id=media_id,
            )
            create_feature(feature)

        print("Features created")
        return True

    def select_diverse_poses(
        self,
        poses,
        build_embedding_fn,
        frame_lengths,  # cumulative end-frames for each clip,
        threshold=5.0,  # distance threshold for "difference"
        min_per_clip=2,  # minimum number of poses per clip
    ):
        """
        Selects diverse poses based on a distance threshold, 
        but also ensures each clip has at least 'min_per_clip' selections.

        Args:
            poses (list or np.ndarray): shape (N, 13, 2)
            build_embedding_fn (callable): returns a (d,)-dim embedding from a pose
            frame_lengths (list of int): cumulative sum of frames 
                indicating the end of each video clip. 
                For example, if clip1 has 1336 frames, clip2 has 380, etc., 
                frame_lengths might look like [1336, 1716, 2855, 3249, ...].
            threshold (float): distance threshold to consider a pose "unique enough."
            min_per_clip (int): enforce this minimum selection per clip.
            order (str): 'original' => sequential from 0..N-1,
                        'random' => shuffled order.

        Returns:
            selected_indices (list of int): indices of poses selected
        """
        # Build embeddings for all poses
        all_embeddings = [build_embedding_fn(pose) for pose in poses]
        all_embeddings = np.array(all_embeddings)  # shape (N, d)
        N = len(poses)

        # Determine clip boundaries
        # frame_lengths is cumulative end-frames, e.g. [1336, 1716, 2855, 3249, ...]
        # So clip i starts at frame_lengths[i-1] (or 0 if i=0),
        # and ends at frame_lengths[i]-1.
        clip_index_for_frame = np.zeros(N, dtype=int)
        current_clip = 0
        start_frame = 0
        for i, end_frame in enumerate(frame_lengths):
            # e.g. frames from [start_frame .. end_frame-1] belong to clip i
            clip_index_for_frame[start_frame:end_frame] = i
            start_frame = end_frame
        total_clips = len(frame_lengths)

        indices = np.arange(N)

        selected_indices = []
        selected_embeddings = []

        # Main selection loop: distance-threshold approach
        for idx in indices:
            emb = all_embeddings[idx]

            if not selected_embeddings:
                selected_indices.append(idx)
                selected_embeddings.append(emb)
            else:
                dists = [np.linalg.norm(emb - se)
                         for se in selected_embeddings]
                if all(d > threshold for d in dists):
                    selected_indices.append(idx)
                    selected_embeddings.append(emb)

        # Post-process each clip to ensure min_per_clip
        selected_indices_set = set(selected_indices)

        # Build a list of frames that belong to each clip
        clip_frames = [[] for _ in range(total_clips)]
        for frame_idx in range(N):
            cidx = clip_index_for_frame[frame_idx]
            clip_frames[cidx].append(frame_idx)

        # Check how many are selected in each clip
        for clip_id in range(total_clips):
            frames_in_clip = clip_frames[clip_id]
            already_selected = [
                f for f in frames_in_clip if f in selected_indices_set]

            if len(already_selected) < min_per_clip:
                needed = min_per_clip - len(already_selected)
                unselected = [
                    f for f in frames_in_clip if f not in selected_indices_set]
                unselected.sort()

                to_add = unselected[:needed]
                # if len(to_add) < needed:
                #     print(
                #         f"⚠️  Clip {clip_id} only has {len(already_selected) + len(to_add)} poses available; can't meet min_per_clip={min_per_clip}")

                selected_indices.extend(to_add)
                selected_indices_set.update(to_add)
                for f in to_add:
                    selected_embeddings.append(all_embeddings[f])

        selected_indices = sorted(selected_indices)

        return selected_indices

    def create_binary_pose_embeddings(self) -> bool:
        #  Retrieve all binary pose files from the db

        query = text(
            "SELECT * FROM feature WHERE feature_type = 'pose-binary-extracted' LIMIT 10")
        result = DataAccessObject().fetch_all(query)
        df = pd.DataFrame(result)

        for i, row in self.tqdm(df.iterrows(), leave=False, total=len(df), position=1, desc='Clips'):
            # Set embedding_33 and embedding_size and update the feature
            pose = row.data['keypoints']

            # create embedding with build_pose_vector_with_flip
            embedding = build_pose_vector_with_flip(pose)
            query = text(
                "UPDATE feature SET embedding_33 = :embedding, embedding_size = 33 WHERE feature_id = :feature_id")
            DataAccessObject().execute_query(
                query, {'embedding': embedding.tolist(), 'feature_id': row.feature_id})
            print(row.feature_id, embedding)
        return True

    def create_projection(self, df: pd.DataFrame) -> bool:
        """ Create the projection for all clips in the dataframe. """
        for i, row in self.tqdm(df.iterrows(), leave=False, total=len(df), position=1, desc='Clips'):
            self.create_single_projection(row)
        return True
