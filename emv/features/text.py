
import pandas as pd

from typing import List, Optional, Any, Dict, Tuple, Union
from pathlib import Path
from collections import defaultdict
from scenedetect.frame_timecode import FrameTimecode


import emv.utils

LOG = emv.utils.get_logger()

_model = {
    'name': None,
    'model': None
}

SWISS_CITIES = None
ALL_CITIES = None


def load_model(model_name: str = 'fr_core_news_lg'):
    import spacy
    # spacy.prefer_gpu()
    global _model
    if not _model['name']:
        model_name = 'fr_core_news_lg'
        LOG.info(f'Loading spacy model: {model_name}')
        _model['model'] = spacy.load(model_name)
        _model['name'] = model_name


def run_nlp(text: str, model_name: Optional[str] = None) -> Any:
    global _model
    if not _model['name']:
        load_model(model_name)

    doc = _model['model'](text)
    return doc


def extract_locations(text: str, model_name: Optional[str] = None) -> List[str]:
    doc = run_nlp(text, model_name)
    # Fix Zurich
    res = set([ent.text for ent in doc.ents if ent.label_ == 'LOC'])
    if not res:
        return None
    return res


def get_swiss_cities() -> Dict:
    import geonamescache

    global SWISS_CITIES, ALL_CITIES
    if not SWISS_CITIES:
        gc = geonamescache.GeonamesCache(min_city_population=500)
        cities = gc.get_cities()
        ALL_CITIES = cities

        SWISS_CITIES = {}
        for k, v in cities.items():
            if v["countrycode"] == "CH":
                SWISS_CITIES[v["name"]] = {
                    "lat": v["latitude"], "lon": v["longitude"], 'geoid': v['geonameid']}
                for alt in v["alternatenames"]:
                    SWISS_CITIES[alt] = {
                        "lat": v["latitude"], "lon": v["longitude"], 'geoid': v['geonameid']}

        # Monkey patch cities
        SWISS_CITIES['Baal'] = SWISS_CITIES['Basel']
        SWISS_CITIES['Saint-Gal'] = SWISS_CITIES['St. Gallen']
        SWISS_CITIES['St-Maurice'] = SWISS_CITIES['Saint-Maurice']
    return SWISS_CITIES


def find_locations(transcript: List[Dict],
                   filtered_locations: Optional[Dict] = None,
                   model_name: Optional[str] = None) -> Optional[List[Dict]]:
    """Add locations using entity recognition to transcript"""

    LOG.debug(f'Find locations in transcript')
    if not filtered_locations:
        filtered_locations = get_swiss_cities()

    global ALL_CITIES
    for sent in transcript:
        locs = extract_locations(sent['t'], model_name)
        if not locs:
            continue

        cities = list(filter(lambda x: x in filtered_locations, locs))
        if not cities:
            continue

        norm_cities = []
        for city in cities:
            geoid = filtered_locations[city]['geoid']
            # Normalize city name
            c = ALL_CITIES[str(geoid)]['name']
            norm_cities.append(c)

        sent['locations'] = ' | '.join(norm_cities)

    return transcript


def merge_continous_sentences(transcript: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(transcript) == 0:
        return transcript
    merged_transcript = [transcript[0]]
    for i in range(1, len(transcript)):
        if transcript[i]['sid'] == merged_transcript[-1]['sid']:
            merged_transcript[-1]['e'] = transcript[i]['e']
            merged_transcript[-1]['t'] += ' ' + transcript[i]['t']
            if 'locations' in transcript[i]:
                if 'locations' in merged_transcript[-1]:
                    # Get unique locations
                    locations = set(
                        merged_transcript[-1]['locations'].split(' | ') + transcript[i]['locations'].split(' | '))
                    merged_transcript[-1]['locations'] = ' | '.join(locations)
                else:
                    merged_transcript[-1]['locations'] = transcript[i]['locations']
        else:
            merged_transcript.append(transcript[i])
    return merged_transcript


def timecodes_from_transcript(transcript: List[Dict[str, str]], framerate: int = 25,
                              min_seconds: float = 10,
                              extend_duration: float = 0,
                              min_words: int = 15,
                              location_only: bool = False) -> Optional[Tuple[List[Tuple[FrameTimecode, FrameTimecode]], List[int]]]:
    if not transcript:
        return None

    timecodes = []
    saved_idx = []
    for i, sent in enumerate(transcript):
        if location_only and 'locations' not in sent:
            continue

        word_count = len(sent['t'].split(' '))
        if word_count < min_words:
            continue

        start = FrameTimecode(float(sent['s']), framerate)
        end = FrameTimecode(float(sent['e']), framerate)
        duration = end - start
        if duration.get_seconds() < min_seconds:
            continue

        if duration.get_seconds() < extend_duration:
            delta = extend_duration - duration.get_seconds()
            start -= delta
            if start.get_seconds() < 0:
                start = FrameTimecode(0, framerate)

        saved_idx.append(i)
        try:
            entities = sent['entities']
        except KeyError:
            entities = []
        timecodes.append((start, end, sent['t'], entities))
    return timecodes, saved_idx


def enrich_scenes_with_transcript(scenes: Dict, ori_transcript: List[Dict]) -> Dict:
    transcript = ori_transcript.copy()

    for i, s in scenes['clips'].items():
        scenes['clips'][i]['transcript'] = ''

        scene_start = float(s['start'])
        scene_end = float(s['end'])

        last_tidx = 0
        for tidx, t in enumerate(transcript):
            t_start = float(t['s'])
            t_end = float(t['e'])

            # if 'locations' in t:
            #     print(t_start, t_end, ' -- ', scene_start, scene_end)

            if t_start > scene_end:  # overshooting timestamp
                last_tidx = tidx
                break

            if t_end <= scene_end or t_start >= scene_start:
                scenes['clips'][i]['transcript'] += t['t'] + ' '
                if 'locations' in t:
                    if 'locations' not in scenes['clips'][i]:
                        scenes['clips'][i]['locations'] = set(
                            t['locations'].split(' | '))
                    else:
                        scenes['clips'][i]['locations'] |= set(
                            t['locations'].split(' | '))

        # start from last_tidx
        transcript = transcript[last_tidx:]

    for i, s in scenes['clips'].items():
        if 'locations' in scenes['clips'][i]:
            scenes['clips'][i]['locations'] = ' | '.join(
                scenes['clips'][i]['locations'])

    return scenes


def find_locations_all_transcripts(transcripts: Dict[str, Dict], filtered_locations: Optional[Dict] = None, model_name: Optional[str] = None) -> Dict[str, Dict]:
    """Add locations using entity recognition to all transcripts"""
    res = {}
    for k, v in transcripts.items():
        ts = find_locations(v, filtered_locations, model_name)
        if ts:
            res[k] = ts
    return res


def build_location_df(transcripts: Dict[str, Dict]) -> pd.DataFrame:
    """Get media with fixed multilang locations"""
    import geonamescache
    gc = geonamescache.GeonamesCache(min_city_population=500)
    all_cities = gc.get_cities()
    swiss_cities = get_swiss_cities()

    res = []
    for k, v in transcripts.items():
        for sent in v:
            if 'locations' in sent:
                cities = sent['locations'].split(' | ')
                for city in cities:
                    if not city:
                        continue

                    geoid = swiss_cities[city]['geoid']
                    # Normalize city name
                    c = all_cities[str(geoid)]['name']
                    res.append({
                        'mediaId': k,
                        'location': c,
                        's': sent['s'],
                        'e': sent['e'],
                        't': sent['t'],
                        'sid': sent['sid']
                    })

    ts = pd.DataFrame.from_records(res)
    ts.set_index('mediaId', inplace=True)
    return ts


class TextEmbedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextEmbedder, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        # Lazy loading of PyTorch and transformers
        import torch
        from transformers import CamembertModel, CamembertTokenizer

        self.tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-large')
        self.model = CamembertModel.from_pretrained('camembert/camembert-large')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def encode(self, texts: List[str]) -> List:
        return self.create_embeddings(texts)

    def create_embeddings(self, texts: Union[str, List[str]]) -> List:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for paragraph in texts:
            # Lazy import of torch within the method
            import torch
            inputs = self.tokenizer(paragraph, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state

            emb = last_hidden_states.mean(dim=1).to('cpu')
            embeddings.append(list(emb[0].numpy()))

        return embeddings
    

def create_embeddings(texts: Union[str, List[str]]) -> List:
    embedder = TextEmbedder()
    return embedder.create_embeddings(texts)
