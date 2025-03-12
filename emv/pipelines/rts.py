import os
import sys
import pandas as pd
import shutil
import xml.etree.ElementTree as ET

import emv.utils
import emv.io.media
import emv.features.audio
import emv.features.text
from emv.io.media import extract_audio, get_frame_number, save_clips_images
from emv.db.queries import create_or_update_media, get_feature_by_media_id_and_type, create_feature, update_feature, get_feature_wout_embedding_1024, get_media_clips_by_type_sub_type, get_media_by_parent_id_and_types
from emv.features.text import run_nlp, timecodes_from_transcript, create_embeddings
from emv.pipelines.utils import get_media_info
from emv.storage.storage import get_storage_client
from emv.io.media import get_media_info as generate_media_info

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from emv.settings import RTS_ROOT_FOLDER
from emv.pipelines.base import Pipeline, get_hash
from emv.db import queries
from emv.api.models import Media, Feature
from emv.db.dao import DataAccessObject

LOG = emv.utils.get_logger()
LOG.setLevel('INFO')

TRANSCRIPT_CLIP_MIN_SECONDS = 10
TRANSCRIPT_CLIP_MIN_WORDS = 15
TRANSCRIPT_CLIP_EXTEND_DURATION = 0
TRANSCRIPT_CLIP_NUM_IMAGES = 3

SCENE_EXPORT_NAME = 'scenes.json'

RTS_METADATA = os.path.join(RTS_ROOT_FOLDER, 'metadata')
RTS_LOCAL_VIDEOS = os.path.join(RTS_ROOT_FOLDER + 'archive')


class PipelineRTS(Pipeline):
    library_name: str = 'rts'

    def ingest(self, df: pd.DataFrame,
               merge_continous_sentences: bool = True,
               compute_transcript: bool = True,
               compute_clips: bool = True,
               force_media: bool = False,
               force_trans: bool = False,
               force_clips: bool = False,
               prefix_path: str = None) -> bool:
        """
        Ingest the RTS metadata and video files.

        Parameters:
            df (pd.DataFrame): RTS metadata dataframe which is expected to have the following columns:
                - mediaId (str): unique identifier for the video
                # TODO: add the rest of the columns

            merge_continous_sentences (bool): merge continous sentences in the transcript
            compute_transcript (bool): compute the transcript
            compute_clips (bool): compute the clips
            force_media (bool): force the media to be reprocessed
            force_trans (bool): force the transcript to be reprocessed
            force_clips (bool): force the clips to be reprocessed
        """

        DataAccessObject().set_user_id(1)
        total_duration = df.mediaDuration.sum()

        if prefix_path is None:
            prefix_path = self.library['prefix_path']

        with self.tqdm(total=total_duration, file=sys.stdout) as pbar:
            for _, row in df.iterrows():
                try:
                    input_file_path = os.path.join(
                        prefix_path, row['mediaFolderPath'])
                    print(input_file_path)
                    self.ingest_single_video(input_file_path, merge_continous_sentences,
                                             compute_transcript, compute_clips, force_media, force_trans, force_clips)
                except TypeError as e:
                    import traceback
                    LOG.error(
                        f"Error processing media: {row['mediaFolderPath']}")
                    LOG.error(traceback.format_exc())
                pbar.update(row.mediaDuration)

    def ingest_single_video(self,
                            input_file_path: str,
                            merge_continous_sentences: bool = True,
                            compute_transcript: bool = False,
                            compute_clips: bool = False,
                            force_media: bool = False,
                            force_trans: bool = False,
                            force_clips: bool = False) -> dict:
        """ Ingest all clips from a single video. """

        archive_media_id = emv.utils.get_media_id(input_file_path)
        media_id = f"{self.library_name}-{archive_media_id}"
        original_path = os.path.join(
            input_file_path, f'{archive_media_id}.mp4')
        export_path = os.path.join(input_file_path, 'export')

        media = queries.get_media_by_id(archive_media_id)
        if media is None or force_media:
            # LOG.info("Processing media: %s", archive_media_id)
            media_info = get_media_info(original_path, input_file_path)
            metadata = {}

            # Create or update media entry in the database
            video = Media(**{
                'media_id': media_id,
                'original_path': original_path,
                'original_id': archive_media_id,
                'media_path': original_path,
                'media_type': "video",
                'media_info': media_info,
                'sub_type': "clip",
                'size': media_info['filesize'],
                'metadata': metadata,
                'library_id': self.library_id,
                'hash': get_hash(original_path),
                'parent_id': "",
                'start_ts': 0,
                'end_ts': media_info['duration'],
                'start_frame': get_frame_number(0, media_info['video']['framerate']),
                'end_frame': get_frame_number(media_info['duration'], media_info['video']['framerate']),
                'frame_rate': media_info['video']['framerate'],
            })
            create_or_update_media(video)

        if compute_transcript:
            # Check if an audio file already exists
            audio_path = os.path.join(
                input_file_path, f'{archive_media_id}_audio.mp3')
            audio_path_mp4 = os.path.join(
                input_file_path, f'{archive_media_id}_audio.m4a')
            if not os.path.exists(audio_path):
                # Extract audio from video
                extract_audio(original_path, export_path)
            try:
                self.transcript = emv.features.audio.transcribe_media(
                    audio_path)
            except RuntimeError as e:
                self.transcript = emv.features.audio.transcribe_media(
                    audio_path_mp4)

            for i, transcript in enumerate(self.transcript):
                entities = run_nlp(transcript['t'])
                entities = [(e.text, e.label_) for e in entities.ents]
                if entities:
                    self.transcript[i]['entities'] = entities

            feature_type = 'transcript+ner'
            # Get feature from the db to see if it already exists
            feature = queries.get_feature_by_media_id_and_type(
                media_id, feature_type)

            new_feature = Feature(
                feature_type=feature_type,
                version="1",
                model_name='whisperx+spacy',
                model_params={
                    'spacy-ner': 'fr_core_news_lg',
                },
                data={'transcript': self.transcript},
                media_id=media_id,
            )
            if feature:
                queries.update_feature(feature['feature_id'], new_feature)
            else:
                queries.create_feature(new_feature)

        if compute_clips:
            sentences = self.transcript
            if merge_continous_sentences:
                sentences = emv.features.text.merge_continous_sentences(
                    self.transcript)

            timecodes = timecodes_from_transcript(sentences, min_seconds=8)
            if not timecodes or not timecodes[0]:
                # skipping because no long enough clip was found
                return
            num_images = 3
            self.clip_infos = save_clips_images(
                timecodes[0], original_path, input_file_path, 'import', num_images, 'M')

            for key in self.clip_infos['clips']:
                self.upload_clips(original_path, archive_media_id, media_id,
                                  self.clip_infos['clips'][key], compute_transcript)

            # Save images
            for key in self.clip_infos['image_binaries']:
                self.upload_clip_images(
                    archive_media_id, media_id, key, self.clip_infos['image_binaries'][key])

    def upload_clips(self, original_path, archive_media_id, media_id, clip_info, compute_transcript):
        media_path = f"videos/{archive_media_id}/{clip_info['clip_id']}.mp4"
        media_info = self.trim_upload_media(
            original_path, media_path, clip_info['start'], clip_info['end'])

        metadata = {}
        clip_media_id = "rts-" + clip_info['clip_id']
        clip = Media(**{
            'media_id': clip_media_id,
            'original_path': original_path,
            'original_id': archive_media_id,
            'media_path': media_path,
            'media_type': "video",
            'media_info': media_info,
            'sub_type': "clip",
            'size': media_info['filesize'],
            'metadata': metadata,
            'library_id': self.library_id,
            'hash': get_hash(media_path),
            'parent_id': media_id,
            'start_ts': clip_info['start'],
            'end_ts': clip_info['end'],
            'start_frame': clip_info['start_frame'],
            'end_frame': clip_info['end_frame'],
            'frame_rate': media_info['video']['framerate'],
        })
        create_or_update_media(clip)

        if compute_transcript:
            feature_type = 'transcript+ner'
            # Get feature from the db to see if it already exists
            feature = queries.get_feature_by_media_id_and_type(
                clip_media_id, feature_type)

            new_feature = Feature(
                feature_type=feature_type,
                version="1",
                model_name='whisperx+spacy',
                model_params={
                    'spacy-ner': 'fr_core_news_lg',
                },
                data={'transcript': clip_info['sentences'],
                      'entities': clip_info['entities']},
                media_id=clip_media_id,
            )
            if feature:
                queries.update_feature(feature['feature_id'], new_feature)
            else:
                queries.create_feature(new_feature)

        return media_info

    def extract_audio_from_clips(self, batch_no: int, number_of_batches: int, force: bool = False) -> bool:
        """ Extract audio from all clips in the dataframe. This assumes that all clips are already ingested and uses the db and s3 to retrieve the clips. """
        DataAccessObject().set_user_id(1)
        # get all media objects that are clips. e.g. media_type = 'video' and sub_type = 'clip'
        print("Extracting audio from clips")
        clips = get_media_clips_by_type_sub_type(
            self.library_id, 'video', 'clip', 'media_id, media_path, start_ts, end_ts, start_frame, end_frame')
        print(f"Found {len(clips)} clips")

        #  assume the clips are processed in 10 batches and the batch_no is the current batch
        batch_size = len(clips) // number_of_batches
        start = batch_no * batch_size
        end = (batch_no + 1) * batch_size
        clips = clips[start:end]

        for clip in self.tqdm(clips, position=1, desc='Clips'):
            self.extract_audio_from_clip(clip, force)
        return True

    def extract_audio_from_clip(self, media: dict, force: bool = False) -> bool:
        """ Extract audio from a single clip. """

        # skip clips that are too long
        # if media['end_ts'] - media['start_ts'] > 60 * 60:
        #     return

        # check the database if there is already an audio media object
        audio_id = f"{media['media_id']}_audio"
        audio = queries.get_media_by_id(audio_id)
        if audio and not force:
            # print("alraedy exists")
            return False

        #  get the audio from the s3 storage
        stream = get_storage_client().get_bytes(
            self.library_name, media['media_path'])

        # extract audio from the video
        audio_path = os.path.join('/tmp', f'{media["media_id"]}_audio.mp4')
        if not stream:
            # print(f"Could not get stream for {media['media_id']}")
            return False

        with open(audio_path, 'wb') as f:
            f.write(stream)

        # Extract audio from the video
        extracted_audio_path = os.path.join(
            '/tmp', f'{media["media_id"]}_extracted_audio.ogg')
        try:
            extract_audio(audio_path, extracted_audio_path,
                          output_format='ogg')
        except Exception as e:
            # print(f"Error extracting audio: {e}")
            return False

        # Create or update the audio media object
        audio_info = generate_media_info(extracted_audio_path)
        media_id_short = media['media_id'].split('-')[1]
        media_path = f"audios/{media_id_short}/{audio_id}.ogg"
        audio_media = Media(
            media_id=audio_id,
            original_path=extracted_audio_path,
            original_id=media['media_id'],
            media_path=media_path,
            media_type="audio",
            media_info=audio_info,
            sub_type="audio",
            size=audio_info['filesize'],
            metadata={},
            library_id=self.library_id,
            hash=get_hash(extracted_audio_path),
            parent_id=media['media_id'],
            start_ts=media['start_ts'],
            end_ts=media['end_ts'],
            start_frame=media['start_frame'],
            end_frame=media['end_frame'],
            frame_rate=0,
        )
        create_or_update_media(audio_media)

        # upload the audio to the s3 storage
        with open(extracted_audio_path, 'rb') as f:
            get_storage_client().upload(
                self.library_name, audio_media.media_path, f)

        # remove tmp files
        os.remove(audio_path)
        os.remove(extracted_audio_path)

        return True

    def upload_clip_images(self, archive_media_id, media_id, key, img_binary):

        for bin_key in img_binary:
            img_path = f"images/{archive_media_id}/{bin_key}"
            self.store.upload(self.library_name, img_path,
                              img_binary[bin_key].getvalue())

            current_clip = self.clip_infos['clips'][key]
            image_id = img_path.split('/')[-1].split('.')[0]

            screenshot = Media(**{
                'media_id': "rts-" + image_id,
                'original_path': img_path,
                'original_id': archive_media_id,
                'media_path': img_path,
                'media_type': "image",
                'media_info': {},
                'sub_type': "screenshot",
                'size': -1,
                'metadata': {},
                'library_id': self.library_id,
                'hash': get_hash(img_path),
                'parent_id': "rts-" + key,
                'start_ts': current_clip['start'],
                'end_ts': current_clip['end'],
                'start_frame': current_clip['start_frame'],
                'end_frame': current_clip['end_frame'],
                'frame_rate': -1,
            })
            create_or_update_media(screenshot)

    def extract_thumbnails(self, batch_no: int, number_of_batches: int, force: bool = False) -> bool:
        """ Extract thumbnails from all clips in the dataframe. This assumes that all clips are already ingested and uses the db and s3 to retrieve the clips. """
        DataAccessObject().set_user_id(1)
        # get all media objects that are clips. e.g. media_type = 'video' and sub_type = 'clip'
        print("Extracting thumbnails from clips")
        clips = get_media_clips_by_type_sub_type(
            self.library_id, 'video', 'clip', 'media_id, media_path, start_ts, end_ts, start_frame, end_frame')
        print(f"Found {len(clips)} clips")

        #  assume the clips are processed in 10 batches and the batch_no is the current batch
        batch_size = len(clips) // number_of_batches
        start = batch_no * batch_size
        end = (batch_no + 1) * batch_size
        clips = clips[start:end]

        for clip in self.tqdm(clips, position=1, desc='Clips'):
            self.extract_thumbnails_from_clip(clip, force)
        return True

    def extract_thumbnails_from_clip(self, media: dict, force: bool = False) -> bool:
        """ Extract thumbnails from a single clip. """

        # check the database if there is already an audio media object
        thumbnail = get_media_by_parent_id_and_types(
            media['media_id'], 'image', 'screenshot')
        if thumbnail and not force:
            return False

        #  get the video from the s3 storage
        stream = get_storage_client().get_bytes(
            self.library_name, media['media_path'])

        # upload_clip_images
        img_binaries = save_clips_images(
            [(media['start_ts'], media['end_ts'])], stream, 'import', 1, 'M')['image_binaries']

        for key in img_binaries:
            self.upload_clip_images(
                media['original_id'], media['media_id'], key, img_binaries[key])

        return True

    def preprocess(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass

    def generate_embeddings(self) -> pd.DataFrame:
        # iterate over the db a few features at a time and generate embeddings
        no_processed = 0
        with self.tqdm() as pbar:
            while features := get_feature_wout_embedding_1024(feature_type='transcript+ner', limit=100):
                texts = []

                for row in features:
                    try:
                        texts.append(
                            "".join([x['t'] for x in row['data']['transcript']]))
                    except TypeError:
                        # There are two feature types, full clip or split
                        texts.append(row['data']['transcript'])

                embeddings = create_embeddings(texts)

                for i, row in enumerate(features):
                    updated_feature = dict(row)
                    updated_feature['embedding_1024'] = list(embeddings[i])
                    updated_feature['model_params']['embedding_1024'] = "camembert/camembert-large"
                    updated_feature['embedding_size'] = 1024
                    feature = Feature(**updated_feature)
                    queries.update_feature(feature.feature_id, feature)

                pbar.update(len(features))


def get_raw_video_audio_parts(media_folder: str) -> Tuple[str, str]:
    def extract_base_urls(xml_string):
        root = ET.fromstring(xml_string)
        ns = {'': 'urn:mpeg:dash:schema:mpd:2011'}
        base_urls = []
        for adaptation_set in root.findall(".//AdaptationSet", ns):
            for representation in adaptation_set.findall(".//Representation", ns):
                base_url = representation.find("BaseURL", ns).text
                bitrate = int(representation.get("bandwidth"))
                mimeType = representation.get(
                    'mimeType', 'null/mp4').split('/')[0]
                base_urls.append((base_url, mimeType, bitrate))
        base_urls = sorted(base_urls, key=lambda x: x[2], reverse=True)[:2]
        return base_urls

    xml_string = emv.utils.read_mpd_file(media_folder)
    urls = extract_base_urls(xml_string)

    video = None
    audio = None
    for u in urls:
        if u[1] == 'video':
            video = os.path.join(media_folder, u[0])
        else:
            audio = os.path.join(media_folder, u[0])
    return video, audio
