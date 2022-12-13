import os
import av
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rts.utils

LOG = rts.utils.get_logger()


@rts.utils.timeit
# noinspection PyUnresolvedReferences
def to_wav(in_path: str, out_path: str = None, sample_rate: int = 48000) -> str:
    """Arbitrary media files to wav"""
    if out_path is None:
        out_path = os.path.splitext(in_path)[0] + '.wav'
    with av.open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        in_stream.thread_type = "AUTO"
        with av.open(out_path, 'w', 'wav') as out_container:
            out_stream = out_container.add_stream(
                'pcm_s16le',
                rate=sample_rate,
                layout='mono'
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    return out_path


@rts.utils.timeit
def to_mp3(in_path: str, out_path: str = None, bitrate: str = '') -> str:
    if out_path is None:
        out_path = os.path.splitext(in_path)[0] + '.mp3'
    with av.open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        in_stream.thread_type = "AUTO"
        with av.open(out_path, 'w', 'mp3') as out_container:
            opts = {}
            if bitrate:
                opts = {
                    'b': bitrate
                }
            out_stream = out_container.add_stream(
                codec_name='mp3',
                options=opts
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    return out_path


@rts.utils.timeit
def remux_audio(in_path: str, out_path: str = None) -> Optional[str]:
    try:
        with av.open(in_path) as in_container:
            in_stream = in_container.streams.audio[0]
            in_stream.thread_type = "AUTO"
            ext = in_stream.codec_context.codec.name  
            out_path = str(Path(out_path).with_suffix(f'.{ext}'))
            with av.open(out_path, 'w') as out_container:
                out_stream = out_container.add_stream(template=in_stream)
                for packet in in_container.demux(in_stream):
                    # We need to skip the "flushing" packets that `demux` generates.
                    if packet.dts is None:
                        continue
                    # We need to assign the packet to the new stream.
                    packet.stream = out_stream
                    out_container.mux(packet)
        return out_path
    except av.AVError as e:
        LOG.error(e)
        return None