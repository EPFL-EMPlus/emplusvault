import os
import sys
import pandas as pd
import shutil
import xml.etree.ElementTree as ET

import emv.utils
import emv.io.media
import emv.features.audio
import emv.features.text
from emv.io.media import extract_audio, get_media_info, get_frame_number, save_clips_images
from emv.db.queries import create_or_update_media, get_feature_by_media_id_and_type, create_feature, update_feature
from emv.features.text import run_nlp, timecodes_from_transcript

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from emv.settings import RTS_ROOT_FOLDER
from emv.pipelines.base import Pipeline, get_hash
from emv.db import queries
from emv.api.models import Media, Feature

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
               force_clips: bool = False) -> bool:
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

        total_duration = df.mediaDuration.sum()
        with self.tqdm(total=total_duration, file=sys.stdout) as pbar:
            for _, row in df.iterrows():
                try:
                    input_file_path = os.path.join(
                        self.library['prefix_path'], row['mediaFolderPath'])
                    self.ingest_single_video(input_file_path, merge_continous_sentences,
                                             compute_transcript, compute_clips, force_media, force_trans, force_clips)
                except TypeError as e:
                    LOG.error(
                        f"Error processing media: {row['mediaFolderPath']}")
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
                'parent_id': -1,
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
                input_file_path, f'{archive_media_id}_audio.m4a')
            if not os.path.exists(audio_path):
                # Extract audio from video
                extract_audio(original_path, export_path)
            self.transcript = emv.features.audio.transcribe_media(audio_path)

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
                version=1,
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
                version=1,
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

    def preprocess(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass


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


def create_optimized_media(media_folder: str, output_folder: str, force: bool = False) -> Dict:
    """
    Process and optimize media from a given media folder and export the results to an output folder.

    Parameters:
    - media_folder (str): The path to the source media folder containing raw video and audio files.
    - output_folder (str): The path to the folder where the optimized media will be saved.
    - force (bool, optional): If set to True, it will overwrite existing optimized files. Default is False.

    Returns:
    - Dict: A dictionary containing paths to the optimized video and audio files. The keys are 'mediaId', 'mediaFolder', 'video', and 'audio' (if available).

    Notes:
    - The function logs the process using LOG.debug and LOG.error for debugging and error messages respectively.
    - The media ID is retrieved from the media folder and is used to name the optimized files.
    - The function maintains the last four parts of the source media folder's hierarchy in the output folder.
    """
    def get_folder_hierarchy(p):
        return '/'.join(p.split('/')[-4:])

    LOG.debug(f'Create optimized media: {media_folder} -> {output_folder}')
    media_id = emv.utils.get_media_id(media_folder)
    export_folder = Path(os.path.join(
        output_folder, get_folder_hierarchy(media_folder)))

    payload = {
        'mediaId': media_id,
        'mediaFolder': str(export_folder)
    }

    v, a = get_raw_video_audio_parts(media_folder)
    video_path = export_folder.joinpath(f'{media_id}.mp4')
    if not video_path.exists() or force:
        os.makedirs(export_folder, exist_ok=True)  # make folder if needed
        ok = emv.io.media.merge_video_audio_files(v, a, video_path)
        if not ok:
            LOG.error(f'Failed to merge video and audio: {media_id}')
            return payload

    payload['video'] = str(video_path)

    audio_path = export_folder.joinpath(f'{media_id}_audio.mp3')
    if not audio_path.exists() or force:
        # audio = emv.io.media.extract_audio(a, audio_path)
        audio = emv.io.media.to_mp3(video_path, audio_path, '128k')
        if not audio:
            return payload

    payload['audio'] = str(audio_path)
    return payload


def extract_clips_from_transcript(
    transcript: List[Dict[str, Any]],
    framerate: int,
    media_folder: str,
    video_path: str,
    merge_continous_sentences: bool = True,
    min_seconds: float = 10,
    extend_duration: float = 0,
    min_words: int = 15,
    num_images: int = 3,
    location_only: bool = False,
    force: bool = False,
) -> Optional[Dict]:

    if not media_folder:
        return None

    if not Path(video_path).exists():
        return None

    clip_path = Path(os.path.join(media_folder, 'clips.json'))
    if clip_path.exists() and not force:
        return emv.utils.obj_from_json(clip_path)

    if merge_continous_sentences:
        transcript = emv.features.text.merge_continous_sentences(transcript)

    timecodes, valid_idx = emv.features.text.timecodes_from_transcript(transcript, framerate,
                                                                       min_seconds, extend_duration, min_words, location_only=location_only)

    if not timecodes:
        LOG.debug(f'No valid timecodes from transcript: {video_path}')
        return None

    clip_infos = emv.io.media.save_clips_images(timecodes, video_path, media_folder,
                                                'clips', num_images, 'M' if merge_continous_sentences else 'C')
    if not clip_infos:
        LOG.debug(f'No clips from transcript: {video_path}')
        return None

    # Cut video to clips
    emv.io.media.cut_video_to_clips(clip_infos, force)

    # Add transcript to scene infos with locations
    for i, (cid, clip) in enumerate(clip_infos['clips'].items()):
        t = transcript[valid_idx[i]]
        clip['text'] = t['t']
        clip['locations'] = t.get('locations', '')
        clip['speaker_id'] = t.get('sid', 0)

    emv.utils.obj_to_json(clip_infos, clip_path)
    return clip_infos


def get_media_info(media_path: str, media_folder: str,
                   metadata: Optional[Dict] = None, force: bool = False) -> Optional[Dict]:
    if not media_folder or not media_path:
        return None

    # Load from disk
    p = Path(os.path.join(media_folder, 'media.json'))
    if p.exists() and not force:
        return emv.utils.obj_from_json(p)

    # Save to disk
    media_info = emv.io.media.get_media_info(media_path)
    if not media_info:
        return None

    if metadata:
        media_info['metadata'] = metadata

    ok = emv.utils.obj_to_json(media_info, p)
    if not ok:
        return None

    return media_info


def transcribe_media(media_folder: str, audio_path: str, force: bool = False) -> List[Dict]:
    # emv.utils.obj_to_json(r, os.path.join(OUTDIR, 'transcript.json'))
    if not media_folder:
        return None

    p = Path(os.path.join(media_folder, 'transcript.json'))
    if not force and p.exists():
        LOG.debug(f'Load transcript from disk: {media_folder}')
        return emv.utils.obj_from_json(p)

    LOG.debug(f'Transcribing media: {audio_path}')
    d = emv.features.audio.transcribe_media(audio_path)
    emv.utils.obj_to_json(d, p)
    # Enrich transcript with location
    transcript = emv.features.text.find_locations(d)
    emv.utils.obj_to_json(transcript, p)
    return transcript


def process_media(input_media_folder: str,
                  global_output_folder: str,
                  metadata: Optional[Dict] = None,
                  merge_continous_sentences: bool = True,
                  compute_transcript: bool = False,
                  compute_clips: bool = False,
                  force_media: bool = False,
                  force_trans: bool = False,
                  force_clips: bool = False
                  ) -> Dict:

    res = {
        'mediaId': emv.utils.get_media_id(input_media_folder),
        'status': 'fail',
        'error': ''
    }

    # try:
    remuxed = create_optimized_media(
        input_media_folder,
        global_output_folder,
        force=force_media
    )
    if not remuxed:
        res['error'] = 'Could not remux file'
        return res

    media_info = get_media_info(remuxed.get('video'), remuxed.get(
        'mediaFolder'), metadata, force_media)
    if not media_info:
        res['error'] = 'Could not get media info'
        return res

    if compute_transcript:
        transcript = transcribe_media(
            remuxed.get('mediaFolder'),
            remuxed.get('audio'),
            force=force_trans
        )

        if not transcript:
            res['error'] = 'Could not transcribe media'
            return res

        if compute_clips:
            # Create clips from transcript's timecodes
            clips = extract_clips_from_transcript(transcript,
                                                  media_info.get(
                                                      'framerate', 25),
                                                  remuxed.get('mediaFolder'),
                                                  remuxed.get('video'),
                                                  merge_continous_sentences=merge_continous_sentences,
                                                  min_seconds=TRANSCRIPT_CLIP_MIN_SECONDS,
                                                  extend_duration=TRANSCRIPT_CLIP_EXTEND_DURATION,
                                                  min_words=TRANSCRIPT_CLIP_MIN_WORDS,
                                                  num_images=TRANSCRIPT_CLIP_NUM_IMAGES,
                                                  force=force_clips
                                                  )
            if not clips:
                res['error'] = 'Could not extract clips from transcript'
                return res
    # except Exception as e:
    #     res['error'] = str(e)
    #     return res

    res['status'] = 'success'
    return res


def simple_process_archive(df: pd.DataFrame,
                           global_output_folder: str,
                           merge_continous_sentences: bool = True,
                           compute_transcript: bool = True,
                           compute_clips: bool = True,
                           force_media: bool = False,
                           force_trans: bool = False,
                           force_clips: bool = False) -> None:

    LOG.setLevel('INFO')
    total_duration = df.mediaDuration.sum()
    with tqdm(total=total_duration, file=sys.stdout) as pbar:
        for _, row in df.iterrows():
            status = process_media(row.mediaFolderPath, global_output_folder, row.to_dict(),
                                   merge_continous_sentences, compute_transcript, compute_clips,
                                   force_media, force_trans, force_clips)
            if status['status'] == 'fail':
                LOG.error(f"{status['mediaId']} - {status['error']}")
            else:
                LOG.info(f"{status['mediaId']} - {status['status']}")
            pbar.update(row.mediaDuration)


def get_one_percent_sample(df: pd.DataFrame,
                           nested_struct: str = '0/0',
                           archive_prefix: str = '/mnt/rts/') -> pd.DataFrame:

    sample = df[df.mediaFolderPath.str.startswith(
        f"{archive_prefix}{nested_struct}")]
    return sample.sort_values(by='mediaFolderPath')


def get_ten_percent_sample(df: pd.DataFrame,
                           nested_struct: str = '0',
                           archive_prefix: str = '/mnt/rts/') -> pd.DataFrame:

    sample = df[df.mediaFolderPath.str.startswith(
        f"{archive_prefix}{nested_struct}")]
    return sample.sort_values(by='mediaFolderPath')


def filter_by_asset_type(df: pd.DataFrame,
                         asset_type='Sujet journal télévisé',
                         nested_struct: str = '',
                         archive_prefix: str = '/mnt/rts/') -> pd.DataFrame:

    sample = df[df.assetType == asset_type].sort_values(by='mediaFolderPath')
    if nested_struct:
        sample = sample[sample.mediaFolderPath.str.startswith(
            f"{archive_prefix}{nested_struct}")]

    return sample


def get_sample_df() -> pd.DataFrame:
    LOG.info('Loading metadata')
    df = emv.utils.dataframe_from_hdf5(RTS_METADATA, 'rts_metadata')
    # return emv.metadata.get_ten_percent_sample(df).sort_values(by='mediaFolderPath')
    return filter_by_asset_type(df, nested_struct='0/0')


def get_aivectors_df() -> pd.DataFrame:
    LOG.info('Loading AI vectors subset')
    df = emv.utils.dataframe_from_hdf5(
        RTS_ROOT_FOLDER + '/metadata', 'rts_aivectors')
    return df


# TODO: REMOVE / refactor all below
def load_all_clips(root_folder: str) -> Dict[str, Dict]:
    clips = {}
    for p in emv.utils.recursive_glob(root_folder, match_name='clips.json'):
        media_id = emv.utils.get_media_id(Path(p).parent)
        clips[media_id] = emv.utils.obj_from_json(p)
    return clips


def load_all_media_info(root_folder: str) -> Dict[str, Dict]:
    media = {}
    for p in emv.utils.recursive_glob(root_folder, match_name='media.json'):
        media_id = emv.utils.get_media_id(Path(p).parent)
        media[media_id] = emv.utils.obj_from_json(p)
    return media


# def create_clip_texture_atlases(df: pd.DataFrame, root_dir: str, name: str, tile_size: int = 64,
#                                 no_border: bool = False, flip: bool = True, format='jpg') -> Optional[Dict]:
#     images_paths = []
#     for media_id, v in df.iterrows():
#         clip_folder = v['clip_folder']
#         filename = v['images'][1] # take middle image
#         image_path = Path(clip_folder) / 'images'/ f'{tile_size}px' / filename
#         if not image_path.exists():
#             # LOG.error(f'No image path for {media_id}')
#             continue
#         images_paths.append(image_path)

#     outdir = os.path.join(root_dir, 'atlases')
#     return emv.io.media.create_square_atlases(images_paths, outdir, name,
#                                               max_tile_size=tile_size,
#                                               no_border=no_border,
#                                               flip=flip,
#                                               format=format)


# def clear_all_clips_and_scenes(clip_df: pd.DataFrame) -> None:
#     for _, row in clip_df.iterrows():
#         media_folder = Path(row['clip_folder']).parent
#         clip_path = media_folder / 'clips.json'
#         if clip_path.exists():
#             clip_path.unlink()
#             shutil.rmtree(row['clip_folder'])

#         scene_path = media_folder / 'scenes.json'
#         if scene_path.exists():
#             scene_path.unlink()
#             scene_folder = scene_path.parent / 'scenes'
#             shutil.rmtree(scene_folder)

#         metadata = media_folder / 'metadata.json'
#         if metadata.exists():
#             metadata.unlink()


# def load_transcript(media_folder: str) -> Optional[Dict]:
#     p = Path(os.path.join(media_folder, 'transcript.json'))
#     if p.exists():
#         return emv.utils.obj_from_json(p)
#     return None

# @emv.utils.timeit
# def extract_detected_scenes(
#     media_folder: str,
#     video_path: str,
#     min_seconds: float = 10,
#     num_images: int = 3,
#     force: bool = False) -> Optional[Dict]:

#     if not media_folder:
#         return None

#     scene_out = Path(os.path.join(media_folder, SCENE_EXPORT_NAME))
#     if scene_out.exists() and not force:
#         res = emv.utils.obj_from_json(scene_out)
#         if res is not None: # empty file error
#             return res

#     scenes = emv.io.media.detect_scenes(video_path)
#     if not scenes:
#         LOG.debug(f'Extracting scenes failed: {video_path}')
#         return None

#     scenes = emv.io.media.filter_scenes(scenes, min_seconds)
#     if not scenes:
#         LOG.debug(f'No scenes longer than {min_seconds} secs')
#         return None

#     scene_infos = emv.io.media.save_clips_images(scenes, video_path, media_folder,
#                                                  'scenes', num_images, 'S')

#     save_scenes(scene_infos, media_folder)
#     return scene_infos

# def save_scenes(scenes: Dict, media_folder: str) -> bool:
#     if not media_folder or not scenes:
#         return False

#     p = Path(os.path.join(media_folder, SCENE_EXPORT_NAME))
#     return emv.utils.obj_to_json(scenes, p)

# def load_scene(media_folder: str) -> Optional[Dict]:
#     p = Path(os.path.join(media_folder, SCENE_EXPORT_NAME))
#     if p.exists():
#         return emv.utils.obj_from_json(p)
#     return None


# def load_all_transcripts(root_folder: str) -> Dict[str, Dict]:
#     transcripts = {}
#     for p in emv.utils.recursive_glob(root_folder, match_name='transcript.json'):
#         media_id = emv.utils.get_media_id(Path(p).parent)
#         transcripts[media_id] = emv.utils.obj_from_json(p)
#     return transcripts


# def merge_location_df_with_metadata(metadata: pd.DataFrame, location: pd.DataFrame) -> pd.DataFrame:
#     fdf = metadata[['mediaFolderPath', 'publishedDate', 'title', 'collection', 'contentType']]
#     location = location.join(fdf, on='mediaId', how='inner')
#     return location
