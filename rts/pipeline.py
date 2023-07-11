import os
import sys
import shutil
import pandas as pd
import xml.etree.ElementTree as ET

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm

import rts.metadata
import rts.utils
import rts.io.media
import rts.features.audio
import rts.features.text

LOG = rts.utils.get_logger()

TRANSCRIPT_CLIP_MIN_SECONDS = 10
TRANSCRIPT_CLIP_MIN_WORDS = 15
TRANSCRIPT_CLIP_EXTEND_DURATION = 0
TRANSCRIPT_CLIP_NUM_IMAGES = 3

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

    LOG.debug(f'Create optimized media: {media_folder} -> {output_folder}')
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

    audio_path = export_folder.joinpath(f'{media_id}_audio.mp3')
    if not audio_path.exists() or force:
        # audio = rts.io.media.extract_audio(a, audio_path)
        audio = rts.io.media.to_mp3(video_path, audio_path, '128k')
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
        return rts.utils.obj_from_json(clip_path)
    
    if merge_continous_sentences:
        transcript = rts.features.text.merge_continous_sentences(transcript)

    timecodes, valid_idx = rts.features.text.timecodes_from_transcript(transcript, framerate, 
            min_seconds, extend_duration, min_words, location_only=location_only)
    
    if not timecodes:
        LOG.debug(f'No valid timecodes from transcript: {video_path}')
        return None

    clip_infos = rts.io.media.save_clips_images(timecodes, video_path, media_folder, 
                                                'clips', num_images, 'M' if merge_continous_sentences else 'C')
    if not clip_infos:
        LOG.debug(f'No clips from transcript: {video_path}')
        return None
    
    # Cut video to clips
    rts.io.media.cut_video_to_clips(clip_infos, force)
        
    # Add transcript to scene infos with locations
    for i, (cid, clip) in enumerate(clip_infos['clips'].items()):
        t = transcript[valid_idx[i]]
        clip['text'] = t['t']
        clip['locations'] = t.get('locations', '')
        clip['speaker_id'] = t.get('sid', 0)

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


def transcribe_media(media_folder: str, audio_path: str, force: bool = False) -> List[Dict]:
    # rts.utils.obj_to_json(r, os.path.join(OUTDIR, 'transcript.json'))
    if not media_folder:
        return None

    p = Path(os.path.join(media_folder, 'transcript.json'))
    if not force and p.exists():
        LOG.debug(f'Load transcript from disk: {media_folder}')
        return rts.utils.obj_from_json(p)
    
    LOG.debug(f'Transcribing media: {audio_path}')
    d = rts.features.audio.transcribe_media(audio_path)
    rts.utils.obj_to_json(d, p)
    # Enrich transcript with location
    transcript = rts.features.text.find_locations(d)
    rts.utils.obj_to_json(transcript, p)
    return transcript


def process_media(input_media_folder: str, global_output_folder: str, 
    metadata: Optional[Dict] = None,
    merge_continous_sentences: bool = True,
    compute_transcript: bool = False,
    compute_clips: bool = False,
    force_media: bool = False,
    force_trans: bool = False,
    force_clips: bool = False
    ) -> Dict:

    res = {
        'mediaId': rts.utils.get_media_id(input_media_folder),
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
    
    media_info = get_media_info(remuxed.get('video'), remuxed.get('mediaFolder'), metadata, force_media)
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
                media_info.get('framerate', 25),
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

