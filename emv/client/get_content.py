import os
import cv2
import requests
import numpy as np
from getpass import getpass

from emv.settings import API_BASE_URL, API_MAX_CALLS, API_USERNAME, API_PASSWORD

headers = None

def authenticate():
    print("Authenticating...")
    data = {
        "grant_type": "password",
        "username": API_USERNAME,
        "password": API_PASSWORD,
    }

    # Authenticate
    response = requests.post(f"{API_BASE_URL}/gettoken", data=data, verify=False)

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



def download_video(media_id):
    fn = f"data/videos/{media_id}.mp4"
    if os.path.exists(fn):
        return fn

    global headers
    if headers is None:
        headers = authenticate()

    response = requests.get(f"{API_BASE_URL}/download/{media_id}", headers=headers, verify=False)
    print(f"{API_BASE_URL}/download/{media_id}")
    if response.status_code != 200:
        print("Download failed!")
        return None
    
    
    with open(fn, "wb") as f:
        f.write(response.content)

    return fn

def get_frame(media_id, frame_number):
    video_path = download_video(media_id)
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frame



def get_features(feature_type, page_size = 100, max_features = 100):
    global headers
    if headers is None:
        headers = authenticate()

    if page_size > API_MAX_CALLS:
        print(f"Page size cannot be larger than {API_MAX_CALLS}. Setting page size to {API_MAX_CALLS}")
        page_size = API_MAX_CALLS

    response = requests.get(f"{API_BASE_URL}/features/{feature_type}", 
                            params = {
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
                                params = {
                                    "page_size": page_size, 
                                    "last_seen_feature_id": last_seen_feature_id
                                }, 
                                headers=headers, 
                                verify=False)
        new_results = response.json()
        if type(new_results) == dict and new_results.get("feature_id", None) is None:
            break
        results += new_results

    print(f"Retrieved {len(results)} poses")

    return results
    