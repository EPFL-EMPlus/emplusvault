import hashlib
import os
import shutil
import logging
import math
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple, Any, Union
from io import BytesIO

import numpy as np
import pandas as pd
import av
import ffmpeg
import cv2
import PIL
import scenedetect

from PIL import Image, ImageOps
from scenedetect import open_video, SceneManager, ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from storage3.utils import StorageException

import rts.utils
from rts.db_settings import BUCKET_NAME
from rts.db.queries import create_atlas, create_media, create_projection
from rts.storage.storage import get_supabase_client
from rts.api.models import AtlasCreate, Media, Projection

LOG = rts.utils.get_logger()

logger = logging.getLogger('pyscenedetect')
logger.setLevel(logging.ERROR)


@rts.utils.timeit
# noinspection PyUnresolvedReferenceskeep_only_ids
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
                    out_vstream = out_container.add_stream(
                        template=in_vstream, rate=true_fps)
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


def get_media_info(media_path: str) -> Dict:
    """Get media info"""

    def format_framerate(framerate: Any) -> str:
        if framerate.denominator == 1:
            return str(framerate.numerator)
        return f'{framerate.numerator / framerate.denominator:.3f}'

    info = {}
    try:
        info['path'] = str(media_path)
        info['filesize'] = os.path.getsize(str(media_path))
        info['human_filesize'] = rts.utils.human_readable_size(
            info['filesize'])
        with av.open(str(media_path)) as container:
            info['duration'] = container.duration / 1000000
            info['video'] = {}
            info['audio'] = {}
            for stream in container.streams:
                if stream.type == 'video':
                    info['video']['codec'] = stream.codec_context.name
                    info['video']['width'] = stream.codec_context.width
                    info['video']['height'] = stream.codec_context.height
                    info['video']['framerate'] = format_framerate(
                        stream.guessed_rate)
                    info['video']['bitrate'] = stream.codec_context.bit_rate
                    info['video']['human_bitrate'] = rts.utils.human_readable_bitrate(
                        stream.codec_context.bit_rate)
                elif stream.type == 'audio':
                    info['audio']['codec'] = stream.codec_context.name
                    info['audio']['channels'] = stream.codec_context.channels
                    info['audio']['sample_rate'] = stream.codec_context.sample_rate
                    info['audio']['bitrate'] = stream.codec_context.bit_rate
                    info['audio']['human_bitrate'] = rts.utils.human_readable_bitrate(
                        stream.codec_context.bit_rate)

            if not info['video']:
                del info['video']
            if not info['audio']:
                del info['audio']

    except av.AVError as e:
        LOG.error(e)
    return info


def trim(input_path: Union[str, Path], output_path: Union[str, Path], start_ts: float, end_ts: float) -> bool:
    input_stream = ffmpeg.input(
        str(input_path), hide_banner=None, loglevel='error')
    vid = (
        input_stream.video
        .trim(start=start_ts, end=end_ts)
        .setpts('PTS-STARTPTS')
    )
    aud = (
        input_stream.audio
        .filter_('atrim', start=start_ts, end=end_ts)
        .filter_('asetpts', 'PTS-STARTPTS')
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joined = ffmpeg.concat(vid, aud, v=1, a=1).node
    output = ffmpeg.output(joined[0], joined[1], str(
        output_path)).overwrite_output()
    try:
        output.run(capture_stdout=False)
        return True
    except ffmpeg.Error as e:
        LOG.error(e)

    return False


def save_image_pyramid(image: Image, out_folder: str, name: str, split_folders: bool = False, base_res: int = 16, depth: int = 6) -> Dict:
    """Save image pyramid"""
    images = {}

    if split_folders:
        ori_folder = Path(out_folder) / 'original'
        os.makedirs(str(ori_folder), exist_ok=True)
        images['original'] = str(Path(ori_folder) / f'{name}.jpg')
    else:
        os.makedirs(str(out_folder), exist_ok=True)
        images['original'] = str(Path(out_folder) / f'{name}.jpg')

    image.save(images['original'])

    for i in range(1, depth):
        size = base_res * (2**i)
        if split_folders:
            tmp = Path(out_folder) / f'{size}px'
            os.makedirs(str(tmp), exist_ok=True)
            filename = f'{name}.jpg'
            images[f'{size}px'] = str(Path(tmp) / filename)
        else:
            filename = f'{name}-{size}px.jpg'
            images[f'{size}px'] = str(Path(out_folder) / filename)
        res = PIL.ImageOps.fit(image, size=(size, size))
        res.save(images[f'{size}px'])
    return images


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


def save_clips_images(timecodes: Any,
                      video_path: str,
                      media_folder: str,
                      out_folder_name: str = 'clips',
                      num_images: int = 3,
                      clip_prefix_id: str = 'S') -> Optional[Dict]:

    def inner():
        # Adapted from https://github.com/Breakthrough/PySceneDetect/blob/master/scenedetect/scene_manager.py
        image_name_template: str = '$VIDEO_NAME-$PREFIX$SCENE_NUMBER-$IMAGE_NUMBER'
        filename_template = Template(image_name_template)
        framerate = timecodes[0][0].framerate
        frame_margin = 1
        scene_num_format = '%0'
        scene_num_format += str(max(3,
                                math.floor(math.log(len(timecodes), 10)) + 1)) + 'd'
        image_num_format = '%0'
        image_num_format += str(math.floor(math.log(num_images, 10)) + 2) + 'd'
        timecode_list = [
            [
                FrameTimecode(int(f), fps=framerate) for f in [
                    # middle frames
                    a[len(a) // 2] if (0 < j < num_images - 1) or num_images == 1

                    # first frame
                    else min(a[0] + frame_margin, a[-1]) if j == 0

                    # last frame
                    else max(a[-1] - frame_margin, a[0])

                    # for each evenly-split array of frames in the scene list
                    for j, a in enumerate(np.array_split(r, num_images))
                ]
            ] for i, r in enumerate([
                # pad ranges to number of images
                r if 1 + \
                r[-1] - r[0] >= num_images else list(r) + [r[-1]] * (num_images - len(r))
                # create range of frames in scene
                for r in (
                    range(start.get_frames(), end.get_frames())
                    # for each scene in scene list
                    for start, end in timecodes)
            ])
        ]
        image_names = []
        resolutions = []
        media_id = video.name
        for i, scene_timecodes in enumerate(timecode_list):
            scene_num = scene_num_format % i
            scene_framenames = []
            for j, image_timecode in enumerate(scene_timecodes):
                video.seek(image_timecode)
                frame_im = video.read().astype(np.uint8)
                if frame_im is None:
                    continue

                filename = '%s' % (filename_template.safe_substitute(
                    VIDEO_NAME=media_id,
                    PREFIX=clip_prefix_id,
                    SCENE_NUMBER=scene_num,
                    IMAGE_NUMBER=image_num_format % j,
                    FRAME_NUMBER=image_timecode.get_frames()))

                fullname = f'{filename}.jpg'
                scene_framenames.append(fullname)

                color_converted = cv2.cvtColor(frame_im, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(color_converted)
                images = save_image_pyramid(
                    im, images_folder, filename, split_folders=True)

                if not resolutions:  # fill available resolutions
                    resolutions = list(images.keys())

            image_names.append(scene_framenames)

        return media_id, image_names, resolutions

    clip_folder = Path(media_folder) / out_folder_name
    images_folder = clip_folder / 'images'
    if images_folder.exists():
        # remove old directory
        shutil.rmtree(str(images_folder))
    os.makedirs(str(images_folder), exist_ok=True)

    try:
        video = open_video(video_path, backend='opencv')
    except OSError as e:
        LOG.error(e)
        return None
    media_id, images_names, resolutions = inner()

    # Format timecodes
    payload = {
        'mediaId': media_id,
        'media_folder': str(media_folder),
        'clip_folder': str(clip_folder),
        'image_resolutions': resolutions,
        'framerate': timecodes[0][0].framerate,
        'clip_count': len(timecodes),
        'clips': {}
    }

    for i, s in enumerate(timecodes):
        dur = s[1] - s[0]
        clip_id = f'{media_id}-{clip_prefix_id}{i:03d}'
        payload['clips'][clip_id] = {
            'clip_id': clip_id,
            'clip_number': i,
            'start': float(s[0].get_seconds()),
            'start_frame': int(s[0].get_frames()),
            'end': float(s[1].get_seconds()),
            'end_frame': int(s[1].get_frames()),
            'duration': float(dur.get_seconds()),
            'duration_frames': int(dur.get_frames()),
            'images': images_names[i],
        }

    return payload


def cut_video_to_clips(clips_payload: Dict, force: bool = False) -> bool:
    has_errors = False
    for clip_id, clip in clips_payload['clips'].items():
        start = clip['start']
        end = clip['end']
        video_folder = Path(clips_payload['clip_folder']) / 'videos'
        clip_path = video_folder / f'{clip_id}.mp4'
        if clip_path.exists() and not force:
            return clip_path

        video_path = Path(
            clips_payload['media_folder']) / f"{clips_payload['mediaId']}.mp4"
        ok = trim(video_path, clip_path, start, end)
        if not ok:
            has_errors = True
            LOG.error(
                f'Error cutting clip {clip_id} from {video_path} to {clip_path}')

    return not has_errors


def create_image_grid(images: List[Image.Image],
                      flip: bool = False,
                      grid_size: Tuple[int, int] = (3, 3),
                      grid_spacing: int = 10,
                      grid_border: int = 10,
                      grid_bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
    """Create a grid of images from a list of image paths"""
    if not images:
        return None

    mode = 'RGB'

    width = grid_size[0] * images[0].width + \
        grid_spacing * (grid_size[0] - 1) + 2 * grid_border
    height = grid_size[1] * images[0].height + \
        grid_spacing * (grid_size[1] - 1) + 2 * grid_border
    grid = Image.new(mode, (width, height), grid_bg_color)
    for i, im in enumerate(images):
        x = (i % grid_size[0]) * (images[0].width + grid_spacing) + grid_border
        y = (i // grid_size[0]) * \
            (images[0].height + grid_spacing) + grid_border
        if flip:
            y = height - (y + images[0].height)
        grid.paste(im, (x, y))
    return grid


def create_atlas_texture(images: List[Image.Image],
                         max_width: int = 4096,
                         max_tile_size: int = 128,
                         square: bool = True,
                         no_border: bool = False,
                         flip: bool = True,
                         keep_only_ids: bool = True,
                         bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Optional[Dict]:

    if not images:
        return None

    fim = images[0]
    tile_size = min(max_tile_size, fim.width)
    rows = max(1, max_width // tile_size)
    cols = math.ceil(len(images) / rows)
    if square:
        cols = rows

    # drop if too many images and square
    # images = [images for i, im in enumerate(images) if i < rows * cols]
    images = images[:rows * cols]

    # if keep_only_ids:
    #     images_path = [Path(im).stem for im in images_path]

    ims = []
    for im in images:
        if im.width > tile_size or im.height > tile_size:
            im = ImageOps.fit(im, size=(tile_size, tile_size))
        if not no_border:
            im = ImageOps.crop(im, 1)  # remove borders
        ims.append(im)

    grid_border = 1
    grid_spacing = 1
    if no_border:
        grid_border = 0
        grid_spacing = 0

    atlas_image = create_image_grid(ims,
                                    flip=flip,
                                    grid_size=(rows, cols),
                                    grid_spacing=grid_spacing, grid_border=grid_border, grid_bg_color=bg_color)

    atlas = {
        'images': images[:rows * cols],
        'image_count': len(images),
        'tile_size': (tile_size, tile_size),
        'atlas_size': (rows * tile_size, cols * tile_size),
        'rows': rows,
        'cols': cols,
        'atlas_image': atlas_image,
    }

    return atlas


def create_square_atlases(atlas_name: str,
                          projection_id: int,
                          images: List[Image.Image],
                          max_tile_size: int = 128,
                          width: int = 4096,
                          no_border: bool = False,
                          flip: bool = True,
                          keep_only_ids: bool = True,
                          format: str = 'jpg',
                          bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
                          ) -> Optional[Dict]:
    if not images:
        return None

    max_tiles_per_atlas = (width // max_tile_size) ** 2
    atlases = {}

    for k, i in enumerate(range(0, len(images), max_tiles_per_atlas)):
        atlas_images = images[i:i + max_tiles_per_atlas]
        atlas = create_atlas_texture(atlas_images, width, max_tile_size,
                                     square=True, no_border=no_border, flip=flip,
                                     keep_only_ids=keep_only_ids, bg_color=bg_color)
        atlas['texture_id'] = k
        atlases[k] = atlas

        # save atlas image and create db entry
        res_image = atlas['atlas_image']
        bytes_io = BytesIO()
        res_image.save(bytes_io, format='PNG')
        binary_image = bytes_io.getvalue()

        get_supabase_client().storage.from_(BUCKET_NAME).upload(
            f"{BUCKET_NAME}/atlas/{atlas_name}_{str(k)}.{format}", binary_image)

        atlas = AtlasCreate(
            projection_id=projection_id,
            atlas_order=k,
            atlas_path=f"{BUCKET_NAME}/atlas/{atlas_name}_{str(k)}.{format}",
            atlas_size=atlas['atlas_size'],
            tile_size=atlas['tile_size'],
            tile_count=atlas['rows'] * atlas['cols'],
            rows=atlas['rows'],
            cols=atlas['cols'],
            tiles_per_atlas=atlas['rows'] * atlas['cols'],
        )
        r = create_atlas(atlas)

    return atlases


def upload_clips(clip_df: pd.DataFrame, data_path: str, bucket_name: str = BUCKET_NAME):

    for index, row in clip_df.iterrows():
        video_path = row['mediaFolderPath'].split("/")[-1]
        video_path = os.path.join(data_path, video_path, 'clips', 'videos')
        # List all files in the directory
        try:
            files = os.listdir(video_path)
        except FileNotFoundError:
            continue

        print(video_path)
        for file in files:
            if file.endswith(".mp4"):
                # print(file)
                file_path = os.path.join(video_path, file)
                # print(file_path)
                try:
                    get_supabase_client().storage.from_(bucket_name).upload(
                        f"{bucket_name}/videos/{file}", file_path)
                except StorageException as e:
                    print(e.args[0]['error'])
                    if e.args[0]['error'] != 'Duplicate':
                        raise e
                media_path = f"{bucket_name}/videos/{file}"
                media = Media(
                    original_path=row['mediaFolderPath'],
                    media_path=media_path,
                    media_type="video",
                    sub_type="clip",
                    size=0,
                    metadata={},
                    library_id=1,
                    hash=hashlib.md5(media_path.encode()).hexdigest(),
                    parent_id=1,
                    start_ts=0,
                    ebd_ts=10,
                    start_frame=0,
                    end_frame=10,
                    frame_rate=30,
                )

                create_media(media)


def upload_images(clip_df: pd.DataFrame, data_path: str, bucket_name: str = BUCKET_NAME, image_size: str = '256px'):
    for index, row in clip_df.iterrows():
        video_path = row['mediaFolderPath'].split("/")[-1]
        video_path = os.path.join(
            data_path, video_path, 'clips', 'images', '256px')
        print(video_path)
        # List all files in the directory
        try:
            files = os.listdir(video_path)
        except FileNotFoundError:
            continue
        # print(video_path)
        for file in files:
            if file.endswith(".jpg"):
                # print(file)
                file_path = os.path.join(video_path, file)
                # print(file_path)
                try:
                    get_supabase_client().storage.from_(bucket_name).upload(
                        f"{bucket_name}/images/{image_size}/{file}", file_path)
                except StorageException as e:
                    print(e.args[0]['error'])
                    if e.args[0]['error'] != 'Duplicate':
                        raise e

                # images need to reference the clip and not the source video
                media_path = f"{bucket_name}/images/{image_size}/{file}"
                media = Media(
                    original_path=row['mediaFolderPath'],
                    media_path=media_path,
                    media_type="image",
                    sub_type="screenshot",
                    size=0,
                    metadata={},
                    library_id=1,
                    hash=hashlib.md5(media_path.encode()).hexdigest(),
                    parent_id=1,
                    start_ts=0,
                    ebd_ts=10,
                    start_frame=0,
                    end_frame=10,
                    frame_rate=30,
                )

                result = create_media(media)
                media_id = result['media_id']


def upload_projection(projection_name: str, version: str, library_id: int, model_name: str,
                      model_params: Dict = {}, data: Dict = {}, dimension: int = 2, atlas_folder_path: str = "",
                      atlas_width: int = 1024, tile_size: int = 256, atlas_count: int = 1, total_tiles: int = 8, tiles_per_atlas: int = 8):
    projection = Projection(
        projection_name=projection_name,
        version=version,
        library_id=library_id,
        model_name=model_name,
        model_params=model_params,
        data=data,
        dimension=dimension,
        atlas_folder_path=atlas_folder_path,
        atlas_width=atlas_width,
        tile_size=tile_size,
        atlas_count=atlas_count,
        total_tiles=total_tiles,
        tiles_per_atlas=tiles_per_atlas,
    )
    return create_projection(projection)
