import os
import shutil
import xml.etree.ElementTree as ET

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import whisper

import rts.utils
import rts.io.media

LOG = rts.utils.get_logger()


_model = {
    'name': None,
    'model': None
}

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


def get_media_id(media_folder: str) -> str:
    media_id = Path(media_folder).name
    return media_id


@rts.utils.timeit
def create_optimized_media(media_folder: str, output_folder: str, force: bool = False) -> Dict:
    def get_folder_hierarchy(p):
        return '/'.join(p.split('/')[-4:])

    media_id = get_media_id(media_folder)
    export_folder = Path(os.path.join(output_folder, get_folder_hierarchy(media_folder)))
    os.makedirs(export_folder, exist_ok=True)

    payload = {
        'media_id': media_id,
        'media_folder': str(export_folder)
    }

    v, a = get_raw_video_audio_parts(media_folder)
    video_path = export_folder.joinpath(f'{media_id}.mp4')
    if not video_path.exists() or force:
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

    media_id = get_media_id(media_folder)
    scenes = rts.io.media.detect_scenes(video_path)
    if not scenes:
        return None

    scenes = rts.io.media.filter_scenes(scenes, min_seconds)
    if not scenes:
        return None

    paths = rts.io.media.save_scenes_images(scenes, video_path, media_folder, num_images)
    scene_infos = rts.io.media.format_scenes(media_id, scenes, paths)
    rts.utils.obj_to_json(scene_infos, os.path.join(media_folder, 'scenes.json'))

    return scene_infos


def transcribe_media(audio_path: str, model_name: Optional[str] = None) -> List[Dict]:
    global _model

    if not model_name:
        if not _model['name']:
            model_name = 'medium'
        else:
            model_name = _model['name']

    # not in cache
    if not _model['name']:
        LOG.info(f'Load model: {model_name}')
        _model['model'] = whisper.load_model(model_name)
        _model['name'] = model_name

    # invalidate model
    if model_name != _model['name']:
        LOG.info(f'Change model: {model_name}')
        _model['model'] = whisper.load_model(model_name)
        _model['name'] = model_name

    res = _model['model'].transcribe(audio_path, language='French')
    
    output = []
    for d in res['segments']:
        output.append({
            't': d['text'].strip().replace('-->', '->'),
            's': f"{d['start']:.2f}",
            'e': f"{d['end']:.2f}"
        })
    return output


def process_media(media_folder: str, force: bool = False):
    # optimize media
    # scene detect
    # extract thumbnails
    # transcribe
    # extract features
    pass