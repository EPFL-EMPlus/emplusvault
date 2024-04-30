import os
import yt_dlp
import pandas as pd

from typing import List, Dict, Optional, Union

import emv.utils
from emv.pipelines.base import Pipeline

from emv.settings import YOUTUBE_ROOT_FOLDER
LOG = emv.utils.get_logger()


class YoutubePipeline(Pipeline):
    library_name = 'YouTube'
    
    def __init__(self, library_name: str = library_name, run_notebook: bool = False):
        super().__init__(library_name, run_notebook)

    def ingest(self, df: pd.DataFrame, force: bool = False, **kwargs) -> bool:
        df = self.preprocess(df)
    
    def preprocess(self, df: pd.DataFrame, force_mp4: bool = False, **kwargs) -> pd.DataFrame:
        # Download videos from YouTube
        df = df.dropna(subset=['video_id'])
        for index, row in df.iterrows():
            video_id = row['video_id']
            output_path = f"{YOUTUBE_ROOT_FOLDER}/{row['video_id']}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Skip if video already exists
            if os.path.exists(output_path):
                df.at[index, 'media_path'] = output_path
                continue
            
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'outtmpl': output_path,
            }
            if force_mp4:
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4'
                }]
                
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(video_id)
            df.at[index, 'media_path'] = output_path
        return df


def get_largest_thumbnail(thumbnails: List[Dict]) -> str:
    # Sort thumbnails by area (height x width) and return the largest one
    if not thumbnails:
        return None
    # print(thumbnails)
    largest = max(thumbnails, key=lambda th: th.get('height', 0) * th.get('width', 0))
    return largest['url']


def format_entries(info: Union[Dict, List[Dict]]) -> List[Dict]:
    if not info:
        return None
    
    res = []
    for e in info.get('entries') or [info]:        
        video_info = {
            'video_id': e.get('id'),
            'video_url': e.get('url'),
            'title': e.get('title'),
            'duration': e.get('duration'),
            'description': e.get('description'),
            'thumbnail_url': get_largest_thumbnail(e.get('thumbnails', [])),
            'view_count': e.get('view_count'),
            'channel_id': e.get('channel_id'),
            'uploader_id': e.get('uploader_id')
        }
        res.append(video_info)
    return res


def fetch_videos_from_playlist(playlist_id: str) -> Optional[pd.DataFrame]:
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'dump_single_json': True,
        # 'playlistend': 10,
        'playlist': playlist_id,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_id, download=False)
        d = format_entries(info)
    if not d:
        return None
    return pd.DataFrame(d)


def fetch_videos_from_channel(channel_id: str):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'dump_single_json': True,
        'channel_id': channel_id,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_id, download=False)
        d = format_entries(info)
    if not d:
        return None
    return pd.DataFrame(d)


def fetch_videos_from_search(query: str):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'dump_single_json': True,
        'query': query,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        d = format_entries(info)
    if not d:
        return None
    return pd.DataFrame(d)


def fetch_videos_from_id_list(video_id_list: List):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'dump_single_json': True,
    }
    d = []
    for video_id in video_id_list:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_id, download=False)
            d.extend(format_entries(info))
    if not d:
        return None
    return pd.DataFrame(d)