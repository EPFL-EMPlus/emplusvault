import os
import sys
import shutil
import pandas as pd
import xml.etree.ElementTree as ET

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

import rts.metadata
import rts.utils
import rts.io.media
import rts.features.audio

LOG = rts.utils.get_logger()


def read_xml_file(file_path: str):
    with open(file_path, 'r') as file:
        xml_string = file.read()
    return xml_string


def read_mpd_file(media_folder: str):
    p = Path(media_folder)
    media_id = p.name
    mpd_file = p / Path(media_id + '.mpd')
    return read_xml_file(str(mpd_file))


def extract_base_urls(xml_string):
    root = ET.fromstring(xml_string)
    ns = {'': 'urn:mpeg:dash:schema:mpd:2011'}
    base_urls = []
    for adaptation_set in root.findall(".//AdaptationSet", ns):
        for representation in adaptation_set.findall(".//Representation", ns):
            base_url = representation.find("BaseURL", ns).text
            bitrate = int(representation.get("bandwidth"))
            mimeType = representation.get('mimeType', 'null/mp4').split('/')[0]
            base_urls.append((base_url, mimeType, bitrate))
    base_urls = sorted(base_urls, key=lambda x: x[2], reverse=True)[:2]
    return base_urls


def get_raw_video_audio_parts(media_folder: str) -> Tuple[str, str]:
    xml_string = read_mpd_file(media_folder)
    urls = extract_base_urls(xml_string)

    video = None
    audio = None
    for u in urls:
        if u[1] == 'video':
            video = os.path.join(media_folder, u[0])
        else:
            audio = os.path.join(media_folder, u[0])
    return video, audio


@rts.utils.timeit
def create_optimized_media(media_folder: str, output_folder: str, force: bool = False) -> Dict:
    def get_folder_hierarchy(p):
        return '/'.join(p.split('/')[-4:])

    media_id = rts.utils.get_media_id(media_folder)
    export_folder = Path(os.path.join(output_folder, get_folder_hierarchy(media_folder)))
    
    payload = {
        'media_id': media_id,
        'media_folder': str(export_folder)
    }

    v, a = get_raw_video_audio_parts(media_folder)
    video_path = export_folder.joinpath(f'{media_id}.mp4')
    if not video_path.exists() or force:
        os.makedirs(export_folder, exist_ok=True)  # make folder if needed
        ok = rts.io.media.merge_video_audio_files(v, a, video_path)
        if not ok:
            return payload

    payload['video'] = str(video_path)

    # # Copy video file for now
    # dst_path = export_folder.joinpath(f'{media_id}_video.mp4')
    # if not dst_path.exists() or force:
    #     try:
    #         shutil.copy(v, dst_path)
    #     except (PermissionError, shutil.SameFileError) as e:
    #         LOG.error(e)
    #         return False

    audio_path = export_folder.joinpath(f'{media_id}_audio.m4a')
    if not audio_path.exists() or force:
        audio = rts.io.media.extract_audio(a, audio_path)
        if not audio:
            return payload
    
    payload['audio'] = str(audio_path)
    return payload


@rts.utils.timeit
def extract_scenes(
    media_folder: str,
    video_path: str, 
    min_seconds: float = 10, 
    num_images: int = 3,
    force: bool = False) -> Optional[Dict]:

    if not media_folder:
        return None

    scene_out = Path(os.path.join(media_folder, 'scenes.json'))

    if scene_out.exists() and not force:
        return rts.utils.obj_from_json(scene_out)

    media_id = rts.utils.get_media_id(media_folder)
    scenes = rts.io.media.detect_scenes(video_path)
    if not scenes:
        LOG.error(f'Extract scene failed: {video_path}')
        return None

    scenes = rts.io.media.filter_scenes(scenes, min_seconds)
    if not scenes:
        LOG.warning(f'No scenes longer than {min_seconds} secs')
        return None

    paths = rts.io.media.save_scenes_images(scenes, video_path, media_folder, num_images)
    scene_infos = rts.io.media.format_scenes(media_id, scenes, paths)
    rts.utils.obj_to_json(scene_infos, os.path.join(media_folder, 'scenes.json'))

    return scene_infos


def save_metadata(meta: Dict, media_folder: str, force: bool = False) -> bool:
    if not media_folder:
        return False

    p = Path(os.path.join(media_folder, 'metadata.json'))
    if not p.exists() or force:
        return rts.utils.obj_to_json(meta, p)
    return True


def process_media(input_media_folder: str, global_output_folder: str, 
    metadata: Optional[Dict] = None,
    min_seconds: float = 5, 
    num_images: int = 3,
    compute_transcript: bool = False,
    force: bool = False) -> Dict:

    res = {
        'media_id': rts.utils.get_media_id(input_media_folder),
        'status': 'fail',
        'error': ''
    }

    try:
        remuxed = create_optimized_media(
            input_media_folder, 
            global_output_folder, 
            force=force
        )

        if not remuxed:
            res['error'] = 'Could not remux file'
            return res
        
        ok = save_metadata(metadata, remuxed.get('media_folder'), force)
        if not ok:
            res['error'] = 'Could not save metadata'
            return res

        scenes = extract_scenes(
            remuxed.get('media_folder'),
            remuxed.get('video'),
            min_seconds=min_seconds, 
            num_images=num_images,
            force=force
        )

        if not scenes:
            res['error'] = 'Could not extract scenes'
            return res

        if compute_transcript:
            transcript = rts.features.audio.transcribe_media(
                remuxed.get('media_folder'),
                remuxed.get('audio')
            )

            if not transcript:
                res['error'] = 'Could not transcribe media'
                return res

    except Exception as e:
        res['error'] = e.message()
        return res
    
    res['status'] = 'success'
    return res


def simple_process_archive(df: pd.DataFrame,
    global_output_folder: str, 
    min_seconds: float = 5, 
    num_images: int = 3,
    compute_transcript: bool = False,
    force: bool = False) -> None:

    LOG.setLevel('INFO')
    total_duration = df.mediaDuration.sum()
    with tqdm(total=total_duration, file=sys.stdout) as pbar:
        for _, row in df.iterrows():
            status = process_media(row.mediaFolderPath, global_output_folder, row.to_dict(), 
                min_seconds, num_images, compute_transcript, force)
            LOG.info(status)
            pbar.update(row.mediaDuration)


# Create a main function for the module and launch simple_process_archive
if __name__ == '__main__':
    import argparse
    LOCAL_RTS_DATA = "/media/data/rts/"
    METADATA = LOCAL_RTS_DATA + 'metadata'
    LOCAL_VIDEOS = LOCAL_RTS_DATA + 'archive'

    OUTDIR = 'data'

    parser = argparse.ArgumentParser(description='Process media archive')
    parser.add_argument('--min_seconds', type=float, default=5, help='Minimum duration of a scene')
    parser.add_argument('--num_images', type=int, default=3, help='Number of images per scene')
    parser.add_argument('--compute_transcript', action='store_true', help='Compute transcript')
    parser.add_argument('--force', action='store_true', help='Force processing')
    args = parser.parse_args()

    LOG.info('Loading metadata')
    df = rts.metadata.load_metadata_hdf5(METADATA, 'rts_metadata')

    LOG.info('Get sample df')
    sample_df = rts.metadata.get_one_percent_sample(df)

    LOG.info('Process archive')
    simple_process_archive(sample_df[:1000], LOCAL_VIDEOS, args.min_seconds, args.num_images, args.compute_transcript, args.force)

