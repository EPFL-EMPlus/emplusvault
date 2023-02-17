import os
import io
import zipfile
import orjson
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import rts.utils


LOG = rts.utils.get_logger()


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


def load_metadata_hdf5(outdir: str, name: str) -> pd.DataFrame:
    df = pd.read_hdf(os.path.join(outdir, f'{name}.hdf5'), key='df')
    df.drop_duplicates(subset=['mediaId'], inplace=True)
    df.set_index('mediaId', inplace=True)
    return df


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


def create_clips_df(root_folder: str) -> pd.DataFrame:
    clips = load_all_clips(root_folder)

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
    return df


def create_clip_texture_atlases(df: pd.DataFrame, root_dir: str, tile_size: int = 64, format='jpg') -> Optional[Dict]:
    images_paths = []
    for media_id, v in df.iterrows():
        clip_folder = v['clip_folder']
        filename = v['images'][1] # take middle image
        image_path = os.path.join(clip_folder, f'{tile_size}px', filename)
        if not image_path:
            LOG.error(f'No image path for {media_id}')
            continue
        images_paths.append(image_path)
    
    outdir = os.path.join(root_dir, 'atlases')
    return rts.io.media.create_square_atlases(images_paths, outdir, max_tile_size=tile_size, format=format)
