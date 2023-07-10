import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import rts.utils
import rts.io.media

LOG = rts.utils.get_logger()

_model = {
    'name': None,
    'model': None
}

def transcribe_media(audio_path: str, model_name: Optional[str] = None) -> List[Dict]:
    import whisper
    global _model
    if not audio_path:
        return None

    if not model_name:
        if not _model['name']:
            model_name = 'medium'
        else:
            model_name = _model['name']

    # not in cache
    if not _model['name']:
        LOG.info(f'Load model: {model_name}')
        _model['model'] = whisper.load_model(model_name)
        _model['name'] = model_name

    # invalidate model
    if model_name != _model['name']:
        LOG.info(f'Change model: {model_name}')
        _model['model'] = whisper.load_model(model_name)
        _model['name'] = model_name

    res = _model['model'].transcribe(audio_path, language='French')
    
    output = []
    for d in res['segments']:
        output.append({
            't': d['text'].strip().replace('-->', '->'),
            's': f"{d['start']:.2f}",
            'e': f"{d['end']:.2f}"
        })
    return output



