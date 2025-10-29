import os
import cv2
import click
import requests
import numpy as np
from typing import List, Dict, Tuple, Union, Optional

from emv.storage.storage import get_storage_client
from emv.settings import API_BASE_URL, API_MAX_CALLS, API_USERNAME, API_PASSWORD

headers = None
storage_client = get_storage_client()


class APIAuthenticationError(RuntimeError):
    """Raised when the API cannot authenticate."""


def _get_auth_payload() -> dict[str, str]:
    if not API_USERNAME or not API_PASSWORD:
        raise APIAuthenticationError("Missing API credentials (set API_USERNAME and API_PASSWORD).")
    return {
        "grant_type": "password",
        "username": API_USERNAME,
        "password": API_PASSWORD,
    }


def authenticate() -> dict[str, str]:
    click.echo("Authenticating...")
    data = _get_auth_payload()

    try:
        response = requests.post(f"{API_BASE_URL}/gettoken", data=data, verify=False, timeout=30)
    except requests.RequestException as exc:
        raise APIAuthenticationError(f"Unable to reach API at {API_BASE_URL}: {exc}") from exc

    if response.status_code != 200:
        summary = ""
        try:
            payload = response.json()
        except ValueError:
            payload = response.text.strip()

        if isinstance(payload, dict):
            summary = payload.get("detail") or payload.get("message") or str(payload)
        elif payload:
            summary = payload[:200]

        raise APIAuthenticationError(
            f"Authentication failed (status {response.status_code}){': ' + summary if summary else ''}"
        )

    try:
        json_response = response.json()
    except ValueError as exc:
        raise APIAuthenticationError("Authentication succeeded but response was not JSON.") from exc

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

    try:
        response = requests.get(f"{API_BASE_URL}/download/{media_id}", headers=headers, verify=False, timeout=60)
    except requests.RequestException as exc:
        print(f"Download request failed: {exc}")
        return None
    print(f"{API_BASE_URL}/download/{media_id}")
    if response.status_code != 200:
        headers = authenticate()  # Refresh token
        try:
            response = requests.get(f"{API_BASE_URL}/download/{media_id}", headers=headers, verify=False, timeout=60)
        except requests.RequestException as exc:
            print(f"Download request failed: {exc}")
            return None
        if response.status_code != 200:
            print(f"Download failed (status {response.status_code}): {response.text[:200]}")
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

    response = requests.get(f"{API_BASE_URL}/features/type/{feature_type}",
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
        response = requests.get(f"{API_BASE_URL}/features/type/{feature_type}",
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


@click.group()
def cli() -> None:
    """Command line utilities for quick EMPLUS Vault API checks."""
    pass


@cli.command("auth")
def auth_command() -> None:
    """Authenticate against the API and show the issued token prefix."""
    try:
        token_headers = authenticate()
    except APIAuthenticationError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)
    token = token_headers.get("Authorization", "").replace("Bearer ", "")
    if token:
        click.echo(f"Authenticated OK (token starts with: {token[:12]}...)")
    else:
        click.echo("Authentication did not return a token", err=True)


@cli.command("features")
@click.option("--feature-type", "-t", required=True, help="Feature type slug to fetch.")
@click.option("--page-size", "-p", default=25, show_default=True, type=int,
              help="Records per request (capped by API_MAX_CALLS).")
@click.option("--max-features", "-m", default=100, show_default=True, type=int,
              help="Maximum number of records to retrieve.")
def features_command(feature_type: str, page_size: int, max_features: int) -> None:
    """Fetch a set of features and report how many were returned."""
    try:
        feats = get_features(feature_type, page_size=page_size, max_features=max_features)
    except APIAuthenticationError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)
    click.echo(f"Received {min(len(feats), max_features)} features (raw total {len(feats)})")


@cli.command("download")
@click.argument("media_id")
def download_command(media_id: str) -> None:
    """Download the media file for the given media id."""
    try:
        path = download_video(media_id)
    except APIAuthenticationError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)
    if path:
        click.echo(f"Saved to {path}")
    else:
        click.echo("Download failed", err=True)


if __name__ == "__main__":
    cli()
