import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path
import emv.utils

LOG = emv.utils.get_logger()
LOG.setLevel('INFO')


TRANSCRIPT_CLIP_MIN_SECONDS = 10
TRANSCRIPT_CLIP_MIN_WORDS = 15
TRANSCRIPT_CLIP_EXTEND_DURATION = 0
TRANSCRIPT_CLIP_NUM_IMAGES = 3

SCENE_EXPORT_NAME = 'scenes.json'


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
