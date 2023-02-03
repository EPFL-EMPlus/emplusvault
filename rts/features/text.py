from typing import List, Optional, Any, Dict
from pathlib import Path

import geonamescache
import spacy
# spacy.prefer_gpu()

import rts.utils

LOG = rts.utils.get_logger()

_model = {
    'name': None,
    'model': None
}

SWISS_CITIES = None


def load_model(model_name: str = 'fr_core_news_lg'):
    global _model
    if not model_name:
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
    res = set([ent.text for ent in doc.ents if ent.label_ == 'LOC'])
    if not res:
        return None
    return res


def get_swiss_cities() -> Dict:
    global SWISS_CITIES
    if not SWISS_CITIES:
        gc = geonamescache.GeonamesCache(min_city_population=500)
        cities = gc.get_cities()

        SWISS_CITIES = {}
        for k, v in cities.items():
            if v["countrycode"] == "CH":
                SWISS_CITIES[v["name"]] = {"lat": v["latitude"], "lon": v["longitude"]}
                for alt in v["alternatenames"]:
                    SWISS_CITIES[alt] = {"lat": v["latitude"], "lon": v["longitude"]}
    return SWISS_CITIES
    

def find_locations(transcript: List[Dict], 
    filtered_locations: Optional[Dict] = None, 
    model_name: Optional[str] = None) -> Optional[List[Dict]]:
    """Add locations using entity recognition to transcript"""
    if not filtered_locations:
        filtered_locations = get_swiss_cities()

    # has_loc = False
    # filtered_ts = []

    for sent in transcript:
        locs = extract_locations(sent['t'], model_name)
        if not locs:
            continue
        
        cities = list(filter(lambda x: x in filtered_locations, locs))
        if not cities:
            continue
        
        # has_loc = True
        sent['locations'] = ' | '.join(cities)
        # filtered_ts.append(sent)

    # if not has_loc:
    #     return None

    return transcript


def find_locations_all_transcripts(transcripts: Dict[str, Dict], filtered_locations: Optional[Dict] = None, model_name: Optional[str] = None) -> Dict[str, Dict]:
    """Add locations using entity recognition to all transcripts"""
    res = {}
    for k, v in transcripts.items():
        ts = find_locations(v, filtered_locations, model_name)
        if ts:
            res[k] = ts
    return res
