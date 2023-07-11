import os
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
import logging

from pathlib import Path
from typing import Dict, List, Optional, Tuple


import rts.utils
import rts.io.media

LOG = rts.utils.get_logger()

_model = {
    'whisper_model': None,
    'align_model': None,
    'align_meta': None,
    'diarize_model': None
}


# def extract_sentences(data, min_duration=0, max_duration=None):
#     """
#     This function extracts continuous sentences per speaker from the input data.
#     It also allows to split the sentences into sub-paragraphs based on their duration.
#     """
#     speaker_data = {}

#     for segment in data:
#         speaker = segment['speaker']

#         if speaker not in speaker_data:
#             speaker_data[speaker] = []
        
#         if len(speaker_data[speaker]) == 0 or segment['start'] - speaker_data[speaker][-1]['end'] > min_duration:
#             # Start a new paragraph
#             speaker_data[speaker].append({
#                 'start': segment['start'],
#                 'end': segment['end'],
#                 'text': segment['text'].strip()  # strip leading and trailing spaces
#             })
#         else:
#             # Continue the current paragraph
#             current_paragraph = speaker_data[speaker][-1]
#             current_paragraph['end'] = segment['end']
#             current_paragraph['text'] += " " + segment['text'].strip()  # strip leading and trailing spaces
            
#             if max_duration is not None and current_paragraph['end'] - current_paragraph['start'] > max_duration:
#                 # Split the current paragraph
#                 current_paragraph['end'] = segment['start']
#                 speaker_data[speaker].append({
#                     'start': segment['start'],
#                     'end': segment['end'],
#                     'text': segment['text'].strip()  # strip leading and trailing spaces
#                 })


#     #         't': d['text'].strip().replace('-->', '->'),
#     #         's': f"{d['start']:.2f}",
#     #         'e': f"{d['end']:.2f}"
#     return speaker_data

def extract_continuous_sentences(data, min_duration=0, max_duration=float('inf')) -> List[Dict]:
    speaker_data = []
    for entry in data:
        speaker = int(entry.get('speaker', 'SPEAKER_0').split('_')[1])
        start_time = entry['start']  # Use 'start' time of the sentence
        end_time = entry['end']  # Use 'end' time of the sentence
        duration = end_time - start_time
        if duration < min_duration or duration > max_duration:
            continue
        sentence = entry['text'].strip()
        if len(speaker_data) > 0 and speaker_data[-1]['sid'] == speaker and speaker_data[-1]['e'] == f"{start_time:.2f}":
            speaker_data[-1]['t'] += ' ' + sentence
            speaker_data[-1]['e'] = f"{end_time:.2f}"
        else:
            speaker_data.append({
                's': f"{start_time:.2f}",
                'e': f"{end_time:.2f}",
                't': sentence,
                'sid': speaker
            })
    return speaker_data


def transcribe_media(audio_path: str, lang: str = 'fr', min_duration=10, max_duration=60) -> List[Dict]:
    # import whisper
    import whisperx
    # global _model
    # if not audio_path:
    #     return None

    # if not model_name:
    #     if not _model['name']:
    #         model_name = 'medium'
    #     else:
    #         model_name = _model['name']

    # # not in cache
    # if not _model['name']:
    #     LOG.info(f'Load model: {model_name}')
    #     _model['model'] = whisper.load_model(model_name)
    #     _model['name'] = model_name

    # # invalidate model
    # if model_name != _model['name']:
    #     LOG.info(f'Change model: {model_name}')
    #     _model['model'] = whisper.load_model(model_name)
    #     _model['name'] = model_name

    # res = _model['model'].transcribe(audio_path, language='French')
    
    # output = []
    # for d in res['segments']:
    #     output.append({
    #         't': d['text'].strip().replace('-->', '->'),
    #         's': f"{d['start']:.2f}",
    #         'e': f"{d['end']:.2f}"
    #     })
    # return output

    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
    logging.getLogger('speechbrain').setLevel(logging.ERROR)

    global _model
    device = "cuda"
    if not audio_path:
        return None
    
    audio_file = str(audio_path)
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    if not _model['whisper_model']:
        LOG.debug(f'Load whisper model: large-v2')
        _model['whisper_model'] = whisperx.load_model("large-v2", device, compute_type=compute_type, language=lang)
        
    if not _model['align_model']:
        _model['align_model'], _model['align_meta'] = whisperx.load_align_model(language_code=lang, device=device)

    if not _model['diarize_model']:
        from dotenv import load_dotenv
        load_dotenv()
        _model['diarize_model'] = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)

    audio = whisperx.load_audio(audio_file)
    result = _model['whisper_model'].transcribe(audio, batch_size=batch_size, language=lang)

    # 2. Align whisper output
    result = whisperx.align(result["segments"], _model['align_model'], _model['align_meta'], audio, device, return_char_alignments=False)
    diarize_segments = _model['diarize_model'](audio_file)
    # diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    return extract_continuous_sentences(result["segments"], min_duration, max_duration)

