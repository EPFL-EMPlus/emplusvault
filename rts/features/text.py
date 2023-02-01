from typing import List, Optional, Any, Dict
from pathlib import Path

import spacy
# spacy.prefer_gpu()

import rts.utils

LOG = rts.utils.get_logger()

_model = {
    'name': None,
    'model': None
}

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


def process_transcript(transcript: List[Dict], model_name: Optional[str] = None) -> Dict:
    """Add locations using entity recognition to transcript"""
    for sent in transcript:
        locs = extract_locations(sent['t'], model_name)
        if locs:
            sent['locations'] = ' | '.join(locs)
            print(sent['locations'])
    return transcript


def process_all_transcripts(transcripts: Dict[str, Dict], model_name: Optional[str] = None) -> Dict[str, Dict]:
    """Add locations using entity recognition to all transcripts"""
    for k, v in transcripts.items():
        transcripts[k] = process_transcript(v, model_name)
    return transcripts