import os
import pandas as pd
import emv.utils
import numpy as np
import io
from pathlib import Path
from emv.storage.storage import get_storage_client
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import av
import cv2

from emv.pipelines.base import Pipeline, get_hash
from emv.io.media import get_media_info, get_frame_number
from emv.api.models import Media
from emv.db.queries import create_media, create_or_update_media, check_media_exists
from emv.db.dao import DataAccessObject

from emv.settings import MJF_ROOT_FOLDER

LOG = emv.utils.get_logger()


MJF_METADATA = MJF_ROOT_FOLDER + 'metadata'
MIF_VIDEOS = MJF_ROOT_FOLDER + 'videos'


class PipelineMJF(Pipeline):
    library_name: str = 'mjf'

    def ingest(self, input_file_path: str, df: pd.DataFrame, force: bool = False) -> bool:
        for i, row in self.tqdm(df.iterrows(), total=len(df)):
            self.ingest_single_video(input_file_path, row, force)

        return True

    def ingest_single_video(self, input_file_path: str, row: pd.Series, force: bool = False) -> bool:
        DataAccessObject().set_user_id(1)
        media_id = f"{self.library_name}-{row.song_id}"

        exists = check_media_exists(media_id)
        if exists and not force:
            return True

        local_path = input_file_path + row.path
        media_info = get_media_info(local_path)

        if not media_info:
            LOG.error(f"Skipping {row.path} (no media info)")
            return False

        s3_path = f"videos/{row.path}"
        # Upload to storage
        with open(os.path.join(input_file_path, row.path), 'rb') as f:
            self.store.upload(
                self.library_name, s3_path, f)

        metadata = {
            'title': row.title,
            'concert_id': row.concert_id,
            'concert_name': row.concert_name,
            'date': row.date,
            'location': row.location,
            'genre': row.genre,
            'top_genre': row.top_genre,
            'musicians': row.musicians,
            'instruments': row.instruments
        }

        clip = Media(**{
            'media_id': media_id,
            'original_path': row.path,
            'original_id': str(row.song_id),
            'media_path': s3_path,
            'media_type': "video",
            'sub_type': "clip",
            'size': media_info['filesize'],
            'metadata': metadata,
            'media_info': media_info,
            'library_id': self.library['library_id'],
            'hash': get_hash(row.path),
            'parent_id': "",
            'start_ts': 0,
            'end_ts': media_info['duration'],
            'start_frame': get_frame_number(0, media_info['video']['framerate']),
            'end_frame': get_frame_number(media_info['duration'], media_info['video']['framerate']),
            'frame_rate': media_info['video']['framerate'],
        })

        try:
            create_or_update_media(clip)
        except IntegrityError as e:
            if "duplicate key value violates unique constraint" in str(e):
                LOG.info(
                    f'UniqueViolation: Duplicate media_id {clip.media_id}')
            else:
                raise e

        LOG.debug(f"Created media {clip.media_id}, {clip.media_path}")
        return True

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def create_thumbnails(self, df: pd.DataFrame) -> pd.DataFrame:
        from facenet_pytorch import MTCNN
        import torch
        from ultralytics import YOLO

        # Initialize the person detection model
        person_model = YOLO('yolov8s.pt')

        # Initialize the face detection model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mtcnn = MTCNN(keep_all=True, device=device)

        for i, row in self.tqdm(df.iterrows(), total=len(df)):

            video_bytes = get_storage_client().get_bytes("mjf", row.media_path)
            best_frame, best_person_detections, best_face_detections = process_video_with_face_person(
                video_bytes, person_model, mtcnn, interval=5)

            # Create the square thumbnail
            thumbnail = create_square_thumbnail(
                best_frame, best_person_detections, best_face_detections)

            _, buffer = cv2.imencode('.jpg', thumbnail)
            thumbnail_bytes = io.BytesIO(buffer.tobytes())

            # Define the S3 path for the thumbnail
            thumbnail_s3_path = f"images/{Path(row.media_path).stem}_thumbnail.jpg"

            # Upload the thumbnail to S3
            self.store.upload(self.library_name,
                              thumbnail_s3_path, thumbnail_bytes)

            orig_folder_name = int(Path(row.original_path).stem)
            #  folder names in the mjf archive are organized in groups of 1000
            folder_name = (orig_folder_name % 1000)

            media_path = f"images/{folder_name}/{orig_folder_name}.jpg"
            screenshot = Media(**{
                'media_id': f"{row.media_id}-thumbnail",
                'original_path': f"{row.media_id}-thumbnail",
                'original_id': row.original_id,
                'media_path': media_path,
                'media_type': "image",
                'media_info': {},
                'sub_type': "screenshot",
                'size': -1,
                'metadata': {},
                'library_id': self.library_id,
                'hash': get_hash(media_path),
                'parent_id': row.media_id,
                'start_ts': -1,
                'end_ts': -1,
                'start_frame': -1,
                'end_frame': -1,
                'frame_rate': -1,
            })

            create_or_update_media(screenshot)


def extract_frames_from_bytes(video_bytes, interval=1.0):
    container = av.open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    fps = float(stream.average_rate)
    if fps == 0 or fps != fps:
        fps = 30  # Default FPS?
    frames = []
    interval_frame_count = max(int(round(fps * interval)), 1)
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx % interval_frame_count == 0:
            img = frame.to_ndarray(format='bgr24')
            frames.append((frame_idx, img))
    container.close()
    return frames


def detect_person(frame, model):
    results = model(frame)[0]
    detections = []
    for r in results.boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class': cls
            })
    return detections


def detect_faces(frame, mtcnn):
    # Detect faces using MTCNN
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)
    detections = []
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob is not None:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': float(prob)
                })
    return detections


def score_frame_face_person(face_detections, person_detections, frame_shape):
    height, width = frame_shape[:2]
    frame_area = width * height

    ideal_face_area_ratio = 0.05
    face_sigma = 0.02

    ideal_person_area_ratio = 0.5
    person_sigma = 0.1

    scores = []
    for face_det in face_detections:
        x1_f, y1_f, x2_f, y2_f = face_det['bbox']
        conf_f = face_det['conf']

        # Face area ratio
        face_area = (x2_f - x1_f) * (y2_f - y1_f)
        face_area_ratio = face_area / frame_area

        # Face area score
        face_area_score = np.exp(- ((face_area_ratio -
                                 ideal_face_area_ratio) ** 2) / (2 * face_sigma ** 2))

        # Face position score (prefer centered faces)
        center_x_f = (x1_f + x2_f) / 2 / width
        center_y_f = (y1_f + y2_f) / 2 / height
        face_position_score = 1 - \
            (abs(center_x_f - 0.5) + abs(center_y_f - 0.5))

        for person_det in person_detections:
            x1_p, y1_p, x2_p, y2_p = person_det['bbox']
            conf_p = person_det['conf']

            # Person area ratio
            person_area = (x2_p - x1_p) * (y2_p - y1_p)
            person_area_ratio = person_area / frame_area

            # Person area score
            person_area_score = np.exp(- ((person_area_ratio -
                                       ideal_person_area_ratio) ** 2) / (2 * person_sigma ** 2))

            # Person position score (prefer centered persons)
            center_x_p = (x1_p + x2_p) / 2 / width
            center_y_p = (y1_p + y2_p) / 2 / height
            person_position_score = 1 - \
                (abs(center_x_p - 0.5) + abs(center_y_p - 0.5))

            # Check if face is within person's bounding box
            face_inside_person = (
                x1_p <= x1_f <= x2_p and x1_p <= x2_f <= x2_p and
                y1_p <= y1_f <= y2_p and y1_p <= y2_f <= y2_p
            )

            if face_inside_person:
                # Total score combines face and person detections
                total_score = (
                    conf_f * face_area_score * face_position_score +
                    conf_p * person_area_score * person_position_score
                )

                scores.append(total_score)
    return max(scores) if scores else 0


def process_video_with_face_person(video_bytes, person_model, mtcnn, interval=1.0):
    frames = extract_frames_from_bytes(video_bytes, interval=interval)
    best_score = 0
    best_frame = None
    best_person_detections = None
    best_face_detections = None

    for frame_idx, frame in frames:
        # Detect persons
        person_detections = detect_person(frame, person_model)

        # Detect faces
        face_detections = detect_faces(frame, mtcnn)

        # Score the frame
        score = score_frame_face_person(
            face_detections, person_detections, frame.shape)
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_person_detections = person_detections
            best_face_detections = face_detections

    return best_frame, best_person_detections, best_face_detections


def create_square_thumbnail(frame, person_detections, face_detections):
    import math
    height, width = frame.shape[:2]
    # If there are no detections, center-crop the frame to a square
    if not person_detections and not face_detections:
        # Center-crop frame to square
        min_dim = min(width, height)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        square_frame = frame[start_y:start_y +
                             min_dim, start_x:start_x + min_dim]
        return square_frame

    # Get all bounding boxes
    boxes = []
    for det in person_detections:
        boxes.append(det['bbox'])
    for det in face_detections:
        boxes.append(det['bbox'])

    # Compute the combined bounding box that includes all detections
    x1 = min([box[0] for box in boxes])
    y1 = min([box[1] for box in boxes])
    x2 = max([box[2] for box in boxes])
    y2 = max([box[3] for box in boxes])

    # Center coordinates of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the size of the square (largest possible within the frame)
    box_width = x2 - x1
    box_height = y2 - y1
    half_size = int(math.ceil(max(box_width, box_height)) / 2)

    # Coordinates of the square
    left = int(center_x - half_size)
    top = int(center_y - half_size)
    right = int(center_x + half_size)
    bottom = int(center_y + half_size)

    # Adjust the square to fit within image boundaries
    if left < 0:
        right += -left
        left = 0
    if right > width:
        left -= (right - width)
        right = width
    if top < 0:
        bottom += -top
        top = 0
    if bottom > height:
        top -= (bottom - height)
        bottom = height

    # Ensure the coordinates are within image boundaries
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    # Final check in case adjustments have introduced a size mismatch
    crop_width = right - left
    crop_height = bottom - top
    if crop_width != crop_height:
        # Adjust to make it square
        min_size = min(crop_width, crop_height)
        right = left + min_size
        bottom = top + min_size

    # Crop the image
    cropped_frame = frame[top:bottom, left:right]
    return cropped_frame
