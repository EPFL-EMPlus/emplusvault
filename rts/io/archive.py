import os
from typing import Dict, List, Optional, Tuple

import rts.utils

LOG = rts.utils.get_logger()


def get_video_audio_parts(media_folder: str) -> Tuple[str, str]:
    media_id = media_folder.split('/')[-1]

    video_postfix = '_track1_dashinit.mp4'
    audio_postfix = '_track2_dashinit.mp4'

    video_path = media_id + video_postfix
    audio_path = media_id + audio_postfix
    video = os.path.join(media_folder, video_path)
    audio = os.path.join(media_folder, audio_path)

    return video, audio



# def remux_audio(in_path: str) -> str