import os
from typing import Dict, List, Optional, Tuple

import rts.utils


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
        rts.utils.dict_to_json(vidx, index_path)
        return vidx
    return vidx