import os
import shutil
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


def get_raw_video_audio_parts(media_folder: str) -> Tuple[str, str]:
    media_id = Path(media_folder).name
    video_postfix = '_track1_dashinit.mp4'
    audio_postfix = '_track2_dashinit.mp4'

    video_path = media_id + video_postfix
    audio_path = media_id + audio_postfix
    video = os.path.join(media_folder, video_path)
    audio = os.path.join(media_folder, audio_path)

    return video, audio


def create_optimized_media(media_folder: str, force: bool = False) -> Optional[str]:
    media_id = Path(media_folder).name
    v, a = get_raw_video_audio_parts(media_folder)
    
    export_folder = Path(media_folder) / 'export'
    export_folder.mkdir(exist_ok=True)

    dst_path = export_folder.joinpath(f'{media_id}_ori.mp4')
    if not dst_path.exists() or force:
        ok = rts.io.media.merge_video_audio_files(v, a, dst_path)
        if not ok:
            return False

    # # Copy video file for now
    # dst_path = export_folder.joinpath(f'{media_id}_video.mp4')
    # if not dst_path.exists() or force:
    #     try:
    #         shutil.copy(v, dst_path)
    #     except (PermissionError, shutil.SameFileError) as e:
    #         LOG.error(e)
    #         return False

    dst_path = export_folder.joinpath(f'{media_id}_audio.m4a')
    if not dst_path.exists() or force:
        audio = rts.io.media.extract_audio(a, dst_path)
        if not audio:
            return False
    
    return str(export_folder)


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
    # extract thumbnails
    # transcribe
    pass