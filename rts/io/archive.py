import os
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import rts.utils
import rts.io.media

LOG = rts.utils.get_logger()


def get_raw_video_audio_parts(media_folder: str) -> Tuple[str, str]:
    media_id = Path(media_folder).name
    video_postfix = '_track1_dashinit.mp4'
    audio_postfix = '_track2_dashinit.mp4'

    video_path = media_id + video_postfix
    audio_path = media_id + audio_postfix
    video = os.path.join(media_folder, video_path)
    audio = os.path.join(media_folder, audio_path)

    return video, audio


def create_optimized_media(media_folder: str, force: bool = False) -> bool:
    media_id = Path(media_folder).name
    v, a = get_raw_video_audio_parts(media_folder)
    
    export_folder = Path(media_folder) / 'export'
    export_folder.mkdir(exist_ok=True)

    # Copy video file for now
    dst_path = export_folder.joinpath(f'{media_id}_video.mp4')
    if not dst_path.exists() or force:
        try:
            shutil.copy(v, dst_path)
        except (PermissionError, shutil.SameFileError) as e:
            LOG.error(e)
            return False

    dst_path = export_folder.joinpath(f'{media_id}_audio.m4a')
    if not dst_path.exists() or force:
        audio = rts.io.media.remux_audio(a, dst_path)
        if not audio:
            return False
    
    return True