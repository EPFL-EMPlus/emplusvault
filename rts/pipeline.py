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
import rts.features.text

LOG = rts.utils.get_logger()

TRANSCRIPT_CLIP_MIN_SECONDS = 3
SCENE_EXPORT_NAME = 'scenes.json'


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


def create_optimized_media(media_folder: str, output_folder: str, force: bool = False) -> Dict:
    def get_folder_hierarchy(p):
        return '/'.join(p.split('/')[-4:])

    media_id = rts.utils.get_media_id(media_folder)
    export_folder = Path(os.path.join(output_folder, get_folder_hierarchy(media_folder)))
    
    payload = {
        'mediaId': media_id,
        'mediaFolder': str(export_folder)
    }

    v, a = get_raw_video_audio_parts(media_folder)
    video_path = export_folder.joinpath(f'{media_id}.mp4')
    if not video_path.exists() or force:
        os.makedirs(export_folder, exist_ok=True)  # make folder if needed
        ok = rts.io.media.merge_video_audio_files(v, a, video_path)
        if not ok:
            LOG.error(f'Failed to merge video and audio: {media_id}')
            return payload

    payload['video'] = str(video_path)

    audio_path = export_folder.joinpath(f'{media_id}_audio.m4a')
    if not audio_path.exists() or force:
        audio = rts.io.media.extract_audio(a, audio_path)
        if not audio:
            return payload
    
    payload['audio'] = str(audio_path)
    return payload


@rts.utils.timeit
def extract_detected_scenes(
    media_folder: str,
    video_path: str, 
    min_seconds: float = 10, 
    num_images: int = 3,
    force: bool = False) -> Optional[Dict]:

    if not media_folder:
        return None

    scene_out = Path(os.path.join(media_folder, SCENE_EXPORT_NAME))
    if scene_out.exists() and not force:
        res = rts.utils.obj_from_json(scene_out)
        if res is not None: # empty file error
            return res

    scenes = rts.io.media.detect_scenes(video_path)
    if not scenes:
        LOG.debug(f'Extracting scenes failed: {video_path}')
        return None

    scenes = rts.io.media.filter_scenes(scenes, min_seconds)
    if not scenes:
        LOG.debug(f'No scenes longer than {min_seconds} secs')
        return None

    scene_infos = rts.io.media.save_clips_images(scenes, video_path, media_folder, 
                                                 'scenes', num_images, 'S')
    
    save_scenes(scene_infos, media_folder)
    return scene_infos


def extract_location_clips_from_transcript(
    transcript: Dict,
    framerate: int,
    media_folder: str,
    video_path: str, 
    min_seconds: float = 10,
    extend_duration: float = 10,
    num_images: int = 3,
    force: bool = False) -> Optional[Dict]:

    if not media_folder:
        return None

    if not Path(video_path).exists():
        return None

    clip_path = Path(os.path.join(media_folder, 'clips.json'))
    if clip_path.exists() and not force:
        return rts.utils.obj_from_json(clip_path)

    timecodes, valid_idx = rts.features.text.timecodes_from_transcript(transcript, framerate, 
            min_seconds, extend_duration, location_only=True)
    
    if not timecodes:
        LOG.debug(f'No valid timecodes from transcript: {video_path}')
        return None

    clip_infos = rts.io.media.save_clips_images(timecodes, video_path, media_folder, 
                                                'clips', num_images, 'L')
    if not clip_infos:
        LOG.debug(f'No clips from transcript: {video_path}')
        return None
        
    # Add transcript to scene infos with locations
    for i, (cid, clip) in enumerate(clip_infos['clips'].items()):
        t = transcript[valid_idx[i]]
        clip['ref_text'] = t['t']
        clip['locations'] = t['locations']

    rts.utils.obj_to_json(clip_infos, clip_path)
    return clip_infos



def save_scenes(scenes: Dict, media_folder: str) -> bool:
    if not media_folder or not scenes:
        return False

    p = Path(os.path.join(media_folder, SCENE_EXPORT_NAME))
    return rts.utils.obj_to_json(scenes, p)


def get_media_info(media_path: str, media_folder: str, 
        metadata: Optional[Dict] = None, force: bool = False) -> Optional[Dict]:
    if not media_folder or not media_path:
        return None
    
    # Load from disk
    p = Path(os.path.join(media_folder, 'media.json'))
    if p.exists() and not force:
        return rts.utils.obj_from_json(p)

    # Save to disk
    media_info = rts.io.media.get_media_info(media_path)
    if not media_info:
        return None

    if metadata:
        media_info['metadata'] = metadata
    
    ok = rts.utils.obj_to_json(media_info, p)
    if not ok:
        return None

    return media_info

def transcribe_media(media_folder: str, audio_path: str, 
    model_name: Optional[str] = None, force: bool = False) -> List[Dict]:

    # rts.utils.obj_to_json(r, os.path.join(OUTDIR, 'transcript.json'))
    if not media_folder:
        return None

    p = Path(os.path.join(media_folder, 'transcript.json'))
    if not force and p.exists():
        return rts.utils.obj_from_json(p)
    
    d = rts.features.audio.transcribe_media(audio_path, model_name)
    # Enrich transcript with location
    transcript = rts.features.text.find_locations(d)
    rts.utils.obj_to_json(transcript, p)
    return transcript


def process_media(input_media_folder: str, global_output_folder: str, 
    metadata: Optional[Dict] = None,
    min_seconds: float = 6, 
    num_images: int = 3,
    compute_scenes: bool = False,
    compute_transcript: bool = False,
    force_media: bool = False,
    force_scene: bool = False,
    force_trans: bool = False) -> Dict:

    res = {
        'mediaId': rts.utils.get_media_id(input_media_folder),
        'status': 'fail',
        'error': ''
    }

    try:
        remuxed = create_optimized_media(
            input_media_folder, 
            global_output_folder, 
            force=force_media
        )
        if not remuxed:
            res['error'] = 'Could not remux file'
            return res
        
        media_info = get_media_info(remuxed.get('video'), remuxed.get('mediaFolder'), metadata, force_media)
        if not media_info:
            res['error'] = 'Could not get media info'
            return res

        if compute_scenes:
            scenes = extract_detected_scenes(
                remuxed.get('mediaFolder'),
                remuxed.get('video'),
                min_seconds=min_seconds, 
                num_images=num_images,
                force=force_scene
            )

            if not scenes:
                res['error'] = 'Could not extract scenes'
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

            # Create clips from transcript's locations
            clips = extract_location_clips_from_transcript(transcript,
                media_info.get('framerate', 25),
                remuxed.get('mediaFolder'),
                remuxed.get('video'),
                min_seconds=TRANSCRIPT_CLIP_MIN_SECONDS,
                extend_duration=min_seconds,
                num_images=num_images,
                force=compute_transcript
            )
            if not clips:
                res['error'] = 'Could not extract clips from transcript'
                return res
            
            # if compute_scenes:
                # scenes = rts.features.text.enrich_scenes_with_transcript(scenes, transcript)
    except Exception as e:
        res['error'] = str(e)
        return res
    
    res['status'] = 'success'
    return res


def simple_process_archive(df: pd.DataFrame,
    global_output_folder: str, 
    min_seconds: float = 6, 
    num_images: int = 3,
    compute_scenes: bool = True,
    compute_transcript: bool = True,
    force_media: bool = False,
    force_scene: bool = False,
    force_trans: bool = True) -> None:

    LOG.setLevel('INFO')
    total_duration = df.mediaDuration.sum()
    with tqdm(total=total_duration, file=sys.stdout) as pbar:
        for _, row in df.iterrows():
            status = process_media(row.mediaFolderPath, global_output_folder, row.to_dict(), 
                min_seconds, num_images, compute_scenes, compute_transcript, force_media, force_scene, force_trans)
            LOG.debug(f"{status['mediaId']} - {status['status']}")
            pbar.update(row.mediaDuration)


def load_transcript(media_folder: str) -> Optional[Dict]:
    p = Path(os.path.join(media_folder, 'transcript.json'))
    if p.exists():
        return rts.utils.obj_from_json(p)
    return None


def load_scene(media_folder: str) -> Optional[Dict]:
    p = Path(os.path.join(media_folder, SCENE_EXPORT_NAME))
    if p.exists():
        return rts.utils.obj_from_json(p)
    return None


def load_all_transcripts(root_folder: str) -> Dict[str, Dict]:
    transcripts = {}
    for p in rts.utils.recursive_glob(root_folder, match_name='transcript.json'):
        media_id = rts.utils.get_media_id(Path(p).parent)
        transcripts[media_id] = rts.utils.obj_from_json(p)
    return transcripts


