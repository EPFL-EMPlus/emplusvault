import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import av
import scenedetect
from scenedetect import open_video, SceneManager, ContentDetector

import rts.utils

LOG = rts.utils.get_logger()

logger = logging.getLogger('pyscenedetect')
logger.setLevel(logging.ERROR)


@rts.utils.timeit
# noinspection PyUnresolvedReferences
def to_wav(in_path: str, out_path: str = None, sample_rate: int = 48000) -> str:
    """Arbitrary media files to wav"""
    if out_path is None:
        out_path = str(Path(out_path).with_suffix('.wav'))
    with av.open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        in_stream.thread_type = "AUTO"
        with av.open(out_path, 'w', 'wav') as out_container:
            out_stream = out_container.add_stream(
                'pcm_s16le',
                rate=sample_rate,
                layout='mono'
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    return out_path


@rts.utils.timeit
def to_mp3(in_path: str, out_path: str = None, bitrate: str = '') -> str:
    if out_path is None:
        out_path = str(Path(out_path).with_suffix('.mp3'))
    with av.open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        in_stream.thread_type = "AUTO"
        with av.open(out_path, 'w', 'mp3') as out_container:
            opts = {}
            if bitrate:
                opts = {
                    'b': bitrate
                }
            out_stream = out_container.add_stream(
                codec_name='mp3',
                options=opts
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    return out_path


@rts.utils.timeit
def extract_audio(in_path: str, out_path: str = None) -> Optional[str]:
    try:
        with av.open(str(in_path)) as in_container:
            in_stream = in_container.streams.audio[0]
            in_stream.thread_type = "AUTO"
            ext = in_stream.codec_context.codec.name
            if ext == 'aac':
                ext = 'm4a'
            out_path = str(Path(out_path).with_suffix(f'.{ext}'))
            with av.open(out_path, 'w') as out_container:
                out_stream = out_container.add_stream(template=in_stream)
                for packet in in_container.demux(in_stream):
                    # We need to skip the "flushing" packets that `demux` generates.
                    if packet.dts is None:
                        continue
                    # We need to assign the packet to the new stream.
                    packet.stream = out_stream
                    out_container.mux(packet)
        return out_path
    except av.AVError as e:
        LOG.error(e)
        return None


def merge_video_audio_files(video_path: str, audio_path: str, out_path: str) -> Optional[str]:
    def demux(in_container, in_stream, out_container, out_stream):
        for packet in in_container.demux(in_stream):
            if packet.dts is None:
                continue
            packet.stream = out_stream
            out_container.mux(packet)
    
    try:
        out_path = str(Path(out_path).with_suffix(f'.mp4'))
        with av.open(str(video_path)) as in_video:
            in_vstream = in_video.streams.video[0]
            with av.open(str(audio_path)) as in_audio:
                in_astream = in_audio.streams.audio[0]
                with av.open(out_path, 'w') as out_container:
                    true_fps = int(in_vstream.guessed_rate)
                    out_vstream = out_container.add_stream(template=in_vstream, rate=true_fps)
                    # Fix missing framerate
                    out_vstream.framerate = str(true_fps)
                    out_astream = out_container.add_stream(template=in_astream)
                    # Copy video
                    demux(in_video, in_vstream, out_container, out_vstream)
                    # Copy audio
                    demux(in_audio, in_astream, out_container, out_astream)
        return out_path
    except av.AVError as e:
        LOG.error(e)
        return None



def detect_scenes(video_path: str) -> Optional[List[Tuple[scenedetect.frame_timecode.FrameTimecode]]]:
    """Detect scenes from video analysis"""
    if not video_path:
        return None
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video)
    # `get_scene_list` returns a list of start/end timecode pairs
    # for each scene that was found.
    scenes = scene_manager.get_scene_list()
    return scenes


def filter_scenes(scenes: Any, min_seconds: float = 10) -> Any:
    return list(filter(lambda s: (s[1] - s[0]).get_seconds() >= min_seconds, scenes))


def save_scenes_images(scenes: Any,
    video_path: str, 
    media_folder: Optional[str] = None,
    num_images: int = 3) -> Dict:

    images_folder = Path(media_folder) / 'scenes'

    if images_folder.exists():
        # remove old directory
        shutil.rmtree(str(images_folder))
    os.makedirs(str(images_folder), exist_ok=True)

    # video = open_video(video_path, framerate=25, backend='pyav')
    video = open_video(video_path)
    res = scenedetect.scene_manager.save_images(scenes, 
      video, num_images=num_images, 
      frame_margin=1, 
      image_extension='jpg', 
      encoder_param=95, 
      image_name_template='$VIDEO_NAME-$SCENE_NUMBER-$IMAGE_NUMBER',
      output_dir=images_folder, show_progress=False)

    return res


def format_scenes(media_id: str, scenes: Any, images_dict: Dict) -> Dict:
    payload = {
        'fps': scenes[0][0].framerate,
        'scenes': {}
    }
    for i, s in enumerate(scenes):
        payload['scenes'][f"{media_id}-{i+1:03d}"] = {
            'images': images_dict[i],
            's': s[0].get_timecode(),
            'e': s[1].get_timecode(),
            'dur': (s[1] - s[0]).get_seconds()
        }
    
    return payload

