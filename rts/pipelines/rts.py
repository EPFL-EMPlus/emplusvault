import os
import io
import sys
import shutil
import zipfile
import orjson
import glob
import pandas as pd
import shutil
import xml.etree.ElementTree as ET

import rts.utils
import rts.io.media
import rts.features.audio
import rts.features.text

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from rts.settings import RTS_LOCAL_DATA


LOG = rts.utils.get_logger()

TRANSCRIPT_CLIP_MIN_SECONDS = 10
TRANSCRIPT_CLIP_MIN_WORDS = 15
TRANSCRIPT_CLIP_EXTEND_DURATION = 0
TRANSCRIPT_CLIP_NUM_IMAGES = 3

SCENE_EXPORT_NAME = 'scenes.json'

RTS_METADATA = RTS_LOCAL_DATA + 'metadata'
LOCAL_RTS_VIDEOS = RTS_LOCAL_DATA + 'archive'


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
    xml_string = rts.utils.read_mpd_file(media_folder)
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




def build_video_folder_index(rts_drive_path: str) -> Dict:
    vidx = {}
    l0 = os.listdir(rts_drive_path)
    for i0 in l0:
        if i0.startswith('lost'):
            continue
        l1 = os.listdir(os.path.join(rts_drive_path, i0))
        for i1 in l1:
            l2 = os.listdir(os.path.join(rts_drive_path, i0, i1))
            for i2 in l2:
                l3 = os.listdir(os.path.join(rts_drive_path, i0, i1, i2))
                for vid in l3:
                    p = os.path.join(rts_drive_path, i0, i1, i2, vid)
                    vidx[vid] = p             
    return vidx


def read_video_folder_index(index_path: str) -> Optional[Dict]:
    return rts.utils.dict_from_json(index_path)


def create_load_video_folder_index(rts_drive_path: str, index_path: str) -> Optional[Dict]:
    vidx = read_video_folder_index(index_path)
    if not vidx:
        vidx = build_video_folder_index(rts_drive_path)
        rts.utils.obj_to_json(vidx, index_path)
        return vidx
    return vidx


def parse_item_sequences(data: Dict, media: Dict) -> Dict:
    seqs = {}
    for id in data.get('idSequence', []):
        seq = {
            'guid': media['guid'],
            'mediaId': media['mediaId'],
            'seqid': id,
            'geo': [],
            'mat': [],
            'pp': []
        }
        seqs[seq['seqid']] = seq

    geo = data.get('VisuelsGEOSequence', [])
    for g in geo:
        seqid, loc = g.split('@')
        seqs[seqid]['geo'].append(loc.lower())

    mat = data.get('VisuelsMATSequence', [])
    for m in mat:
        seqid, mat = m.split('@')
        seqs[seqid]['mat'].append(mat.lower())
    
    pp = data.get('VisuelsPPSequence', [])
    for p in pp:
        seqid, name = p.split('@')
        seqs[seqid]['pp'].append(name.lower())
    
    return seqs


def parse_item(raw: Dict, vidx: Dict) -> Optional[Dict]:
    def find_media_id(support: Dict) -> Tuple[Optional[int], Optional[str]]:
        for i, d in enumerate(support):
            if d.startswith('Z'):
                return i, d
        return 0, None

    try:
        d = raw['response']['docs'][0]
    except IndexError as e:
        # LOG.error(e, raw)
        return None
    
    try:
        seq_idx, media_id = find_media_id(d['idSupport'])
    except KeyError as e:
        # LOG.error(e)
        return None
    
    if not media_id:
        return None

    duration = d['DureeMediaSec'][seq_idx]
    ratio = d['RatioMedia'][seq_idx]
    mediaFolderPath = vidx.get(media_id)

    if not mediaFolderPath:
        return None

    resumeSeq = d.get('ResumeSequence')
    resumeSeq = ' '.join(resumeSeq) if resumeSeq else None

    geoTheme = d.get('ThematiquesGEO')
    geoTheme = '; '.join(geoTheme) if geoTheme else None

    bckType = d.get('idTypeBackground')
    bckType = '; '.join(bckType) if bckType else None

    media = {
        'guid': d['Guid'],
        'mediaId': media_id,
        'mediaFolderPath': vidx.get(media_id),
        'mediaDuration': duration,
        'ratio': ratio,
        'formatResolution': d['DefinitionMedia'][seq_idx],
        'publishedDate': d.get('DatePublication'),
        
        'categoryName': d.get('CategorieAsset'),
        'assetType': d.get('TypeAsset'),
        'contentType': d.get('TypeContenu', [None])[0],
        'backgoundType': bckType,
        'collection': d.get('Collection'),
        
        'publishedBy': d.get('CanalPublication', [None])[0],
        'rights': d.get('SemaphoreDroit'),
        'title': d.get('Titre'),
        'resume': d.get('Resume'),
        'geoTheme': geoTheme,
        'resumeSequence': resumeSeq,
        'created': int(datetime.now().timestamp()),
        'published': int(datetime.strptime(d.get('DatePublication'), "%Y-%m-%dT%H:%M:%S%z").timestamp())
    }

    seqs = parse_item_sequences(d, media)
    media['sequences'] = seqs

    return media


def read_txt_transcript(srtz: zipfile.Path, media: Dict) -> Optional[Dict]:
    try:
        ts = {
            'guid': media['guid'],
            'mediaId': media['mediaId'],
            'version': 1,
            'text': ''
        }
        path = media['guid'] + '_0.txt'
        current = srtz.joinpath(path)
        with current.open('r') as fp:
            wrap = io.TextIOWrapper(fp, encoding='latin-1')
            ts['text'] = wrap.read()
    except Exception as e:
        LOG.error('Read txt ts:', e)
        return None
    return ts


def read_metadata_zippart(root_dir: str, part_num: int, vidx: Dict) -> Dict:
    path = os.path.join(root_dir, f'{part_num}.zip')
    medias = {}
    with zipfile.ZipFile(path, 'r') as z:
        jsons = zipfile.Path(z, f'{part_num}/jsonData/')
        for j in jsons.iterdir():
            with j.open('r') as fp:
                js = orjson.loads(fp.read())
                media = parse_item(js, vidx)
                if media:
                # ts = read_txt_transcript(srts, media)
                # media['ts'] = ts
                    medias[media['guid']] = media

    return medias


def read_all_metadata(root_dir: str, vidx: Dict) -> pd.DataFrame:
    all_d = dict()
    for i in range(0, 10):
        d = read_metadata_zippart(root_dir, i, vidx)
        all_d.update(d)
    
    return pd.DataFrame.from_records(list(all_d.values()))


def metadata_to_hdf5(outdir: str, name: str, df: pd.DataFrame) -> bool:
    try:
        df.to_hdf(os.path.join(outdir, f'{name}.hdf5'), key='df', mode='w')
        return True
    except Exception as e:
        return False


def load_metadata_hdf5(input_dir: str, name: str = 'rts_metadata') -> pd.DataFrame:
    df = pd.read_hdf(os.path.join(input_dir, f'{name}.hdf5'), key='df')
    df.drop_duplicates(subset=['mediaId'], inplace=True)
    df.set_index('mediaId', inplace=True)
    return df


def link_aivectors_to_metadata(metadata: pd.DataFrame, input_dir: str, name: str = 'videos.csv') -> pd.DataFrame:
    def match_feat_to_folder():
        base_path = input_dir + '/videos/'
        files = glob.glob(base_path + '*/*/*/*/features.npz', recursive = False)
        feats = {}
        for file in files:
            f = os.path.relpath(str(Path(file).parent), base_path)
            feat_id = f.split('/')[-1]
            feats[feat_id] = str(f)

        s = pd.Series(feats)
        s.name = 'aivector_path'
        return s

    vf = pd.read_csv(Path(input_dir) / name)
    vf.set_index('umid', inplace=True)
    vf.index.name = 'mediaId'
    vf.rename(columns={'id_result': 'aivector_id'}, inplace=True)
    merged = metadata.join(vf, on='mediaId', how='left')
    merged.dropna(subset=['aivector_id'], inplace=True)

    s = match_feat_to_folder()
    return merged.join(s, on='aivector_id', how='left')


def export_metadata_stats(df: pd.DataFrame, outdir: str) -> bool:
    def export(df: pd.DataFrame, col: str) -> bool:
        d = df.groupby(col)[col].count().sort_values(ascending=False)
        os.makedirs(outdir, exist_ok=True)
        d.to_csv(os.path.join(outdir, f'{col}s.csv'))

    export(df, 'contentType')
    export(df, 'assetType')
    export(df, 'collection')
    export(df, 'categoryName')
    

def get_one_percent_sample(df: pd.DataFrame, 
    nested_struct: str = '0/0', 
    archive_prefix: str = '/mnt/rts/') -> pd.DataFrame:

    sample = df[df.mediaFolderPath.str.startswith(f"{archive_prefix}{nested_struct}")]
    return sample.sort_values(by='mediaFolderPath')


def get_ten_percent_sample(df: pd.DataFrame, 
    nested_struct: str = '0', 
    archive_prefix: str = '/mnt/rts/') -> pd.DataFrame:

    sample = df[df.mediaFolderPath.str.startswith(f"{archive_prefix}{nested_struct}")]
    return sample.sort_values(by='mediaFolderPath')


def filter_by_asset_type(df: pd.DataFrame,
    asset_type = 'Sujet journal télévisé',
    nested_struct: str = '', 
    archive_prefix: str = '/mnt/rts/') -> pd.DataFrame:

    sample = df[df.assetType == asset_type].sort_values(by='mediaFolderPath')
    if nested_struct:
        sample = sample[sample.mediaFolderPath.str.startswith(f"{archive_prefix}{nested_struct}")]

    return sample


def merge_location_df_with_metadata(metadata: pd.DataFrame, location: pd.DataFrame) -> pd.DataFrame:    
    fdf = metadata[['mediaFolderPath', 'publishedDate', 'title', 'collection', 'contentType']]
    location = location.join(fdf, on='mediaId', how='inner')
    return location


def load_all_clips(root_folder: str) -> Dict[str, Dict]:
    clips = {}
    for p in rts.utils.recursive_glob(root_folder, match_name='clips.json'):
        media_id = rts.utils.get_media_id(Path(p).parent)
        clips[media_id] = rts.utils.obj_from_json(p)
    return clips


def load_all_media_info(root_folder: str) -> Dict[str, Dict]:
    media = {}
    for p in rts.utils.recursive_glob(root_folder, match_name='media.json'):
        media_id = rts.utils.get_media_id(Path(p).parent)
        media[media_id] = rts.utils.obj_from_json(p)
    return media


def build_clips_df(archive_folder: str, metadata_folder: str, force: bool = False) -> pd.DataFrame:
    if not force:
        df = rts.utils.dataframe_from_hdf5(metadata_folder, 'rts_clips', silent=True)
        if df is not None:
            return df

    clips = load_all_clips(archive_folder)

    payload = []
    for media_id, v in clips.items():
        clip_folder = v['clip_folder']
        for _, clip in v['clips'].items():
            p = clip.copy()
            p['mediaId'] = media_id
            p['clip_folder'] = clip_folder
            payload.append(p)

    df = pd.DataFrame.from_records(payload)
    df.set_index('mediaId', inplace=True)
    rts.utils.dataframe_to_hdf5(metadata_folder, 'rts_clips', df)
    return df


def build_clip_index(df: pd.DataFrame) -> Dict[str, Dict]:
    index = {}
    for media_id, v in df.iterrows():
        media_folder = Path(v['clip_folder']).parent
        video_path = media_folder / f'{media_id}.mp4'
        p = {
            'mediaId': media_id,
            'clip_folder': v['clip_folder'],
            'media_folder': str(media_folder),
            'media_path': str(video_path),
            'start_frame': v['start_frame'],
            'end_frame': v['end_frame'],
        }
        index[v['clip_id']] = p
    return index


def create_clip_texture_atlases(df: pd.DataFrame, root_dir: str, name: str, tile_size: int = 64, 
                                no_border: bool = False, flip: bool = True, format='jpg') -> Optional[Dict]:
    images_paths = []
    for media_id, v in df.iterrows():
        clip_folder = v['clip_folder']
        filename = v['images'][1] # take middle image
        image_path = Path(clip_folder) / 'images'/ f'{tile_size}px' / filename
        if not image_path.exists():
            # LOG.error(f'No image path for {media_id}')
            continue
        images_paths.append(image_path)
    
    outdir = os.path.join(root_dir, 'atlases')
    return rts.io.media.create_square_atlases(images_paths, outdir, name,
                                              max_tile_size=tile_size, 
                                              no_border=no_border,
                                              flip=flip, 
                                              format=format)


def clear_all_clips_and_scenes(clip_df: pd.DataFrame) -> None:
    for _, row in clip_df.iterrows():
        media_folder = Path(row['clip_folder']).parent
        clip_path = media_folder / 'clips.json'
        if clip_path.exists():
            clip_path.unlink()
            shutil.rmtree(row['clip_folder'])
        
        scene_path = media_folder / 'scenes.json'
        if scene_path.exists():
            scene_path.unlink()
            scene_folder = scene_path.parent / 'scenes'
            shutil.rmtree(scene_folder)

        metadata = media_folder / 'metadata.json'
        if metadata.exists():
            metadata.unlink()


def get_sample_df() -> pd.DataFrame:
    LOG.info('Loading metadata')
    df = load_metadata_hdf5(RTS_METADATA, 'rts_metadata')
    # return rts.metadata.get_ten_percent_sample(df).sort_values(by='mediaFolderPath')
    return filter_by_asset_type(df, nested_struct='0/0')


def get_aivectors_df() -> pd.DataFrame:
    LOG.info('Loading AI vectors subset')
    df = rts.utils.dataframe_from_hdf5(
        RTS_LOCAL_DATA + '/metadata', 'rts_aivectors')
    return df
