import time
import pandas as pd
import emv.utils

from pathlib import Path
from typing import Dict, List, Optional, Union


DEFAULT_FILENAME = 'sdsc_files'

LOG = emv.utils.get_logger()


def extract_people(data: dict) -> dict:
    people = dict()
    for p in data['persons']:
        person = {
            'artist_id': p['id'],
            'name': p['name'],
            'wiki_id': p.get('wikidataItemId', '')
        }
        people[p['id']] = person
    return people


def extract_genres(data: dict) -> dict:
    genres = dict()
    for p in data['genres']:
        genre = {
            'genre_id': p['id'],
            'name': p['name'],
            'parents': p['parents'],
            'children': p['children']
        }
        genres[p['id']] = genre
    return genres


def extract_locations(data: dict) -> dict:
    locations = dict()
    for p in data['locations']:
        loc = {
            'location_id': p['id'],
            'name': p['name']
        }
        locations[p['id']] = loc
    return locations


def extract_instruments(data: dict) -> dict:
    instruments = dict()
    for p in data['instruments']:
        instru = {
            'instrument_id': p['id'],
            'name': p['name'],
            'parents': p['parents'],
            'children': p['children']
        }
        instruments[p['id']] = instru
    return instruments


def extract_concerts(data: dict) -> dict:
    concerts = dict()
    for p in data['concerts']:
        concert = {
            'concert_id': p['id'],
            'name': p['name'],
            'date': p['date'],
            'location_id': p['location'],
            'genre_ids': p['genres'],
            'musicians': []
        }

        for m in p['musicians']:
            d = {
                'person_id': m['person'],
                'instrument_ids': m['instruments']
            }
            concert['musicians'].append(d)
        concerts[p['id']] = concert
    return concerts


def extract_export_date(data: dict) -> str:
    t = time.strptime(data['exportDate'], '%Y-%m-%dT%H:%M:%SZ')
    return time.strftime('%Y-%m-%d', t)


def extract_songs(data: dict) -> dict:
    songs = dict()
    for p in data['files']:
        event = p['concertEvent']
        song_id = event['id']
        media_type = p['contentType']

        song = songs.get(song_id)
        if not song:
            song = {
                'song_id': song_id,
                'concert_id': p['concert'],
                'title': event['title'],
                'is_public': True if event['rights'] == 'public' else False,
                'audio_path': '',
                'video_path': ''
            }

        f = p['fileLocations']
        if not f:
            continue
        # only 1 file location per item
        f = f[0]
        url = f['url']

        # update urls
        if media_type == 'audio':
            song['audio_path'] = url
        else:
            song['video_path'] = url

        songs[song['song_id']] = song

    return songs


class MJFMetadataImporter:
    def __init__(self, metadata_dir: str) -> None:
        self.metadata_dir = metadata_dir
        self.metadata = None
        self.path_to_id = None
        self.path_to_id_public = None
        self.id_to_path = None

    def load_metadata_from_disk(self, filename: str = DEFAULT_FILENAME,
                                force: bool = False) -> bool:

        mfolder = self.metadata_dir
        if not mfolder:
            return False

        mfolder = Path(mfolder)
        pick_path = mfolder / 'metadata.pickle'

        self.metadata = dict()

        # Load from cache
        if not force and pick_path.exists():
            try:
                LOG.debug(f'Try loading metadata from pickle: {pick_path}')
                self.metadata = emv.utils.dict_from_pickle(pick_path)
                return True
            except Exception as e:
                pass

        # Read from raw JSON export
        LOG.debug(f'Try loading metadata from raw JSON {filename}.json')
        concerts_path = str(mfolder / f'{filename}.json')
        data = emv.utils.obj_from_json(concerts_path)
        if not data:
            return False

        self.metadata['concerts'] = extract_concerts(data)
        self.metadata['people'] = extract_people(data)
        self.metadata['instruments'] = extract_instruments(data)
        self.metadata['genres'] = extract_genres(data)
        self.metadata['locations'] = extract_locations(data)
        self.metadata['songs'] = extract_songs(data)
        self.metadata['export_date'] = extract_export_date(data)

        emv.utils.dict_to_pickle(self.metadata, pick_path)

        return True

    def index_songs_by_local_video_path(self) -> bool:
        if not self.metadata:
            LOG.error('Metadata not loaded - call load_metadata_from_disk first')
            return False

        self.path_to_id = dict()
        self.path_to_id_public = dict()
        self.id_to_path = dict()
        for s in self.metadata['songs'].values():
            new_k = emv.utils.remove_path_prefix(s['video_path'], 'file://')
            new_k = new_k[:-4]  # remove ext mkv
            if not new_k:  # some video files are missing
                continue

            self.path_to_id[new_k] = s['song_id']
            self.id_to_path[s['song_id']] = new_k + '.mp4'

            if s['is_public']:
                self.path_to_id_public[new_k] = s['song_id']

        return True

    def get_metadata_by_local_path(self, media_path: Union[Path, str]) -> Optional[Dict]:
        if not self.metadata:
            LOG.error(
                'Metadata not loaded - call load_metadata_from_disk then index_songs_by_local_video_path')
            return None

        media_path = Path(media_path)
        if not media_path.is_file():
            LOG.error(f'{str(media_path)} is not a valid file')
            return None

        prefix = str(media_path.parent.parent) + '/'
        filepath = emv.utils.remove_path_prefix(str(media_path), prefix)
        fileid = filepath.split('.')[0]
        song_id = self.path_to_id.get(fileid)

        if not song_id:
            LOG.error(f'Cannot get song_id for {str(media_path)}')
            return None

        return self.get_metadata_by_song_id(song_id)

    def get_metadata_by_song_id(self, song_id: int) -> Optional[Dict]:
        if not self.metadata:
            LOG.error('Metadata not loaded - call load_metadata_from_disk')
            return None

        song = self.metadata['songs'].get(song_id)
        if not song:
            LOG.error(f'Cannot retrieve song_id {song_id}')
            return None

        concert = self.metadata['concerts'].get(song['concert_id'])
        if not concert:
            LOG.error(
                f"Cannot retrieve concert_id {song['concert_id']}, for song {song_id}")
            return None

        location = self.metadata['locations'].get(concert['location_id'])
        if not location:
            LOG.error(
                f"Cannot retrieve concert location {song['concert_id']}, for song {song_id}")
            return None

        d = {
            'song_id': song['song_id'],
            'concert_id': song['concert_id'],
            'title': song['title'],
            'concert_name': concert['name'],
            'date': concert['date'],
            'location': location['name'],
            'genre': '',
            'top_genre': '',
            'path': self.id_to_path.get(song['song_id'], ''),
        }

        # Retrieve genre
        if concert['genre_ids']:
            main_genre_id = concert['genre_ids'][0]
            genre = self.metadata['genres'].get(main_genre_id)
            if genre:
                d['genre'] = str(genre['name'])
                if genre['parents']:
                    top_parent_genre = self.metadata['genres'].get(
                        genre['parents'][0])
                    d['top_genre'] = str(top_parent_genre['name'])
                else:
                    d['top_genre'] = d['genre']

        # Artists
        artists = []
        instruments = set()
        for t in concert['musicians']:
            p = self.metadata['people'].get(t['person_id'])
            artists.append(str(p['name']))

            for iid in t['instrument_ids']:
                instru = self.metadata['instruments'].get(iid)
                if instru:
                    instruments.add(str(instru['name']))

        d['musicians'] = sorted(artists)
        d['instruments'] = sorted(list(instruments))

        return d

    def iter_all_songs(self) -> Dict:
        if not self.metadata:
            LOG.error('Metadata not loaded - call load_metadata_from_disk')
            return None

        song_ids = sorted(self.metadata['songs'].keys())
        for s in song_ids:
            yield self.get_metadata_by_song_id(s)

    def get_metadata(self, media_path: Union[Path, str]) -> Optional[Dict]:
        return self.get_metadata_by_local_path(media_path)

    def get_public_song_paths(self, mjf_folder: Optional[str] = '/') -> List[Path]:
        paths = []
        if not mjf_folder.endswith('/'):
            mjf_folder += '/'

        for k in self.path_to_id_public.keys():
            d = mjf_folder + k + '.mp4'
            paths.append(Path(d))

        return paths

    def get_song_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame.from_records(self.iter_all_songs())
        return df[df.path != '']


def build_mjf_importer(metadata_dir: str, data_filename: str = DEFAULT_FILENAME, force: bool = False) -> MJFMetadataImporter:
    importer = MJFMetadataImporter(metadata_dir)
    importer.load_metadata_from_disk(data_filename, force)
    importer.index_songs_by_local_video_path()
    return importer
