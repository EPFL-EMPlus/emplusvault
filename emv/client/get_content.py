import os
import cv2
import requests
import numpy as np
from getpass import getpass
from typing import List, Dict, Tuple, Union, Optional

from emv.storage.storage import get_storage_client
from emv.settings import API_BASE_URL, API_MAX_CALLS, API_USERNAME, API_PASSWORD

headers = None
storage_client = get_storage_client()


def authenticate() -> dict[str, str]:
    print("Authenticating...")
    data = {
        "grant_type": "password",
        "username": API_USERNAME,
        "password": API_PASSWORD,
    }

    # Authenticate
    response = requests.post(
        f"{API_BASE_URL}/gettoken", data=data, verify=False)

    if response.status_code != 200:
        print("Authentication failed!")

    json_response = response.json()
    access_token = json_response["access_token"]

    # Headers can be used for further requests
    global headers
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    return headers


def download_video(media_id: str) -> str:
    fn = f"data/videos/{media_id}.mp4"
    if os.path.exists(fn):
        return fn

    global headers
    if headers is None:
        headers = authenticate()

    response = requests.get(
        f"{API_BASE_URL}/download/{media_id}", headers=headers, verify=False)
    print(f"{API_BASE_URL}/download/{media_id}")
    if response.status_code != 200:
        headers = authenticate()  # Refresh token
        response = requests.get(
            f"{API_BASE_URL}/download/{media_id}", headers=headers, verify=False)
        if response.status_code == 200:
            print("Download failed!")
            return None

    with open(fn, "wb") as f:
        f.write(response.content)

    return fn


def get_frame(video_id: str, media_id: str, frame_number: int) -> np.ndarray:
    # Check if frame is already in DB
    frame_path = f'images/{video_id}/{media_id}/pose_frame_{frame_number}.jpg'
    frame_bytes = storage_client.get_bytes("ioc", frame_path)
    if type(frame_bytes) == bytes:
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Otherwise, download video and extract frame
        video_path = download_video("ioc-" + media_id)
        if video_path is None:
            return None

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
    return frame


def get_features(feature_type: str, page_size: int = 100, max_features: int = 100) -> List[dict]:
    global headers
    if headers is None:
        headers = authenticate()

    if page_size > API_MAX_CALLS:
        print(
            f"Page size cannot be larger than {API_MAX_CALLS}. Setting page size to {API_MAX_CALLS}")
        page_size = API_MAX_CALLS

    response = requests.get(f"{API_BASE_URL}/features/{feature_type}",
                            params={
                                "page_size": page_size
                            },
                            headers=headers,
                            verify=False)
    results = response.json()

    if max_features is None:
        max_features = np.inf
    while len(results) < max_features:
        try:
            last_seen_feature_id = response.json()[-1]['feature_id']
        except:
            print(response.json())
            break
        response = requests.get(f"{API_BASE_URL}/features/{feature_type}",
                                params={
                                    "page_size": page_size,
                                    "last_seen_feature_id": last_seen_feature_id
                                },
                                headers=headers,
                                verify=False)
        new_results = response.json()
        if type(new_results) == dict and new_results.get("feature_id", None) is None:
            break
        results += new_results
        print(f"Retrieved {len(results)} features so far...")

    print(f"Retrieved {len(results)} features")

    return results
