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

        local_path = os.path.join(input_file_path, row.path)
        media_info = get_media_info(local_path)
        if not media_info:
            LOG.error(f"Skipping {row.path} (no media info)")
            return False

        s3_path = f"videos/{row.path}"
        # If still needed, upload the entire video to S3:
        with open(local_path, 'rb') as f:
            self.store.upload(self.library_name, s3_path, f)

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

    @staticmethod
    def measure_sharpness(frame_bgr: np.ndarray) -> float:
        """
        Higher return value => sharper image.
        Uses the variance of Laplacian on a grayscale version of frame_bgr.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return lap.var()

    @staticmethod
    def extract_frames_limited_duration(
        local_video_path: str,
        interval: float = 4.0,
        max_duration: float = 120.0,  # Only decode up to 2 minutes
        resize_to=(480, 480)
    ):
        import av

        container = av.open(local_video_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        orig_w = stream.codec_context.width
        orig_h = stream.codec_context.height

        fps = float(stream.average_rate) or 30.0
        frames_per_step = max(int(round(fps * interval)), 1)

        frames = []
        for frame_idx, frame in enumerate(container.decode(stream)):
            # If we exceed max_duration, break
            if frame.time is not None and frame.time > max_duration:
                break

            if frame_idx % frames_per_step == 0:
                img = frame.to_ndarray(format='bgr24')
                if resize_to is not None:
                    img = cv2.resize(
                        img, resize_to, interpolation=cv2.INTER_LINEAR)
                frames.append((frame_idx, img, (orig_w, orig_h),
                              (img.shape[1], img.shape[0])))
        container.close()
        return frames

    @staticmethod
    def decode_single_frame_highres(local_video_path: str, target_frame_idx=0):
        import av

        container = av.open(local_video_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        orig_w = stream.codec_context.width
        orig_h = stream.codec_context.height

        for i, frame in enumerate(container.decode(stream)):
            if i == target_frame_idx:
                img_bgr = frame.to_ndarray(format='bgr24')
                container.close()
                return img_bgr, orig_w, orig_h

        container.close()
        return None, 0, 0

    @staticmethod
    def batch_detect_person_and_faces(frames, person_model, mtcnn):
        # YOLO batch
        batch_imgs = [f[1] for f in frames]  # list of BGR images
        yolo_results = person_model(batch_imgs, verbose=False)  # single call

        person_detections_list = []
        for result in yolo_results:
            person_boxes = []
            for r in result.boxes:
                cls = int(r.cls[0])
                conf = float(r.conf[0])
                if cls == 0:  # 'person'
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    person_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class': cls
                    })
            person_detections_list.append(person_boxes)

        # MTCNN batch
        from PIL import Image
        pil_images = [Image.fromarray(cv2.cvtColor(
            f[1], cv2.COLOR_BGR2RGB)) for f in frames]
        boxes_all, probs_all = mtcnn.detect(pil_images)

        face_detections_list = []
        for boxes_arr, probs_arr in zip(boxes_all, probs_all):
            face_boxes = []
            if boxes_arr is not None:
                for (x1_f, y1_f, x2_f, y2_f), prob_f in zip(boxes_arr, probs_arr):
                    if prob_f is not None:
                        face_boxes.append({
                            'bbox': (int(x1_f), int(y1_f), int(x2_f), int(y2_f)),
                            'conf': float(prob_f)
                        })
            face_detections_list.append(face_boxes)

        return person_detections_list, face_detections_list

    def score_frame_face_person(self, face_dets, person_dets, frame_bgr: np.ndarray):
        """
        Assign a score to a frame based on the face and person detections.
        The higher the score, the better the frame.
        """
        import numpy as np
        height, width = frame_bgr.shape[:2]
        frame_area = width * height

        # Face-person parameters
        ideal_face_area_ratio = 0.05
        face_sigma = 0.02

        ideal_person_area_ratio = 0.5
        person_sigma = 0.1

        face_person_scores = []
        for face_det in face_dets:
            x1_f, y1_f, x2_f, y2_f = face_det['bbox']
            conf_f = face_det['conf']

            face_area = (x2_f - x1_f) * (y2_f - y1_f)
            face_area_ratio = face_area / frame_area
            face_area_score = np.exp(-((face_area_ratio -
                                     ideal_face_area_ratio)**2) / (2*face_sigma**2))

            center_x_f = (x1_f + x2_f) / (2.0 * width)
            center_y_f = (y1_f + y2_f) / (2.0 * height)
            face_position_score = 1.0 - \
                (abs(center_x_f - 0.5) + abs(center_y_f - 0.5))

            for person_det in person_dets:
                x1_p, y1_p, x2_p, y2_p = person_det['bbox']
                conf_p = person_det['conf']

                person_area = (x2_p - x1_p) * (y2_p - y1_p)
                person_area_ratio = person_area / frame_area
                person_area_score = np.exp(-((person_area_ratio -
                                           ideal_person_area_ratio)**2)/(2*person_sigma**2))

                center_x_p = (x1_p + x2_p) / (2.0 * width)
                center_y_p = (y1_p + y2_p) / (2.0 * height)
                person_position_score = 1.0 - \
                    (abs(center_x_p - 0.5) + abs(center_y_p - 0.5))

                # Face inside person's bounding box?
                face_inside_person = (
                    x1_p <= x1_f <= x2_p and x1_p <= x2_f <= x2_p and
                    y1_p <= y1_f <= y2_p and y1_p <= y2_f <= y2_p
                )
                if face_inside_person:
                    combined_score = (conf_f * face_area_score * face_position_score +
                                      conf_p * person_area_score * person_position_score)
                    face_person_scores.append(combined_score)

        face_person_score = max(
            face_person_scores) if face_person_scores else 0.0

        # ---- Sharpness bonus ----
        SHARPNESS_WEIGHT = 0.001  # Tweak this
        sharpness_val = self.measure_sharpness(frame_bgr)
        # Weighted sum
        total_score = face_person_score + (SHARPNESS_WEIGHT * sharpness_val)

        return total_score

    @staticmethod
    def create_square_thumbnail(frame, person_detections, face_detections):
        import math
        height, width = frame.shape[:2]

        if not person_detections and not face_detections:
            # Center-crop to square
            min_dim = min(width, height)
            start_x = (width - min_dim) // 2
            start_y = (height - min_dim) // 2
            return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

        boxes = [det['bbox'] for det in person_detections] + \
            [det['bbox'] for det in face_detections]
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        box_w = (x2 - x1)
        box_h = (y2 - y1)
        half_size = max(box_w, box_h) / 2.0

        # Limit half_size so the square won't exceed image bounds
        max_half_w = width / 2.0
        max_half_h = height / 2.0
        half_size = min(half_size, max_half_w, max_half_h)

        left = center_x - half_size
        right = center_x + half_size
        top = center_y - half_size
        bottom = center_y + half_size

        # Shift if out of bounds
        if left < 0:
            shift = -left
            left += shift
            right += shift
        if right > width:
            shift = right - width
            left -= shift
            right -= shift
        if top < 0:
            shift = -top
            top += shift
            bottom += shift
        if bottom > height:
            shift = bottom - height
            top -= shift
            bottom -= shift

        # Round coords
        left = int(round(left))
        right = int(round(right))
        top = int(round(top))
        bottom = int(round(bottom))

        # Final clamp
        if left < 0:
            left = 0
        if right > width:
            right = width
        if top < 0:
            top = 0
        if bottom > height:
            bottom = height

        crop_w = right - left
        crop_h = bottom - top
        if crop_w != crop_h:
            # Make it square by picking the smaller side
            side = min(crop_w, crop_h)
            right = left + side
            bottom = top + side
            if right > width:
                right = width
                left = right - side
            if bottom > height:
                bottom = height
                top = bottom - side

        return frame[top:bottom, left:right]

    def create_thumbnails(self, input_file_path: str, df: pd.DataFrame, force: bool = False, max_duration=180.0) -> pd.DataFrame:
        """
        - Decode only up to `max_duration` seconds from local disk
        - Batch YOLO + MTCNN at low res
        - Incorporate a sharpness factor in the scoring
        - Re-decode best frame in high res for final square thumbnail
        - Upload thumbnail to S3
        """
        from facenet_pytorch import MTCNN
        import torch
        from ultralytics import YOLO

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        person_model = YOLO('yolov8s.pt')
        mtcnn = MTCNN(keep_all=True, device=device)

        for i, row in self.tqdm(df.iterrows(), total=len(df)):
            # check if thumbnail already exists
            exists = check_media_exists(f"{row.media_id}-thumbnail")
            if exists and not force:
                continue
            # LOG.info(f"Processing thumbnail for {row.media_id}")

            # local path to the video
            local_video_path = os.path.join(
                input_file_path, row.original_path)

            # 1) Extract frames (low res) from only first `max_duration` seconds
            frames = self.extract_frames_limited_duration(
                local_video_path,
                interval=4.0,
                max_duration=max_duration,
                resize_to=(480, 480)
            )
            if not frames:
                LOG.warning(f"No frames extracted for {row.media_id}")
                continue

            # 2) Batch detect persons/faces
            person_dets_list, face_dets_list = self.batch_detect_person_and_faces(
                frames, person_model, mtcnn)

            # 3) Score & pick best
            best_score = 0
            best_index_in_list = 0
            for j, (frame_idx, lowres_frame, (orig_w, orig_h), (lw, lh)) in enumerate(frames):
                p_dets = person_dets_list[j]
                f_dets = face_dets_list[j]
                score = self.score_frame_face_person(
                    f_dets, p_dets, lowres_frame)
                if score > best_score:
                    best_score = score
                    best_index_in_list = j
                # optional early stop if score is high enough
                if best_score > 1.25:
                    break

            best_frame_idx = frames[best_index_in_list][0]
            best_lowres_w, best_lowres_h = frames[best_index_in_list][3]
            best_p_dets_low = person_dets_list[best_index_in_list]
            best_f_dets_low = face_dets_list[best_index_in_list]

            # 4) Re-decode best frame at full resolution
            best_highres_frame, orig_w, orig_h = self.decode_single_frame_highres(
                local_video_path, best_frame_idx)
            if best_highres_frame is None:
                LOG.warning(
                    f"Could not decode high-res frame for {row.media_id}")
                continue

            # 5) Scale bounding boxes from low-res to high-res
            scale_w = float(orig_w) / best_lowres_w
            scale_h = float(orig_h) / best_lowres_h

            def scale_dets(dets, sw, sh):
                out = []
                for d in dets:
                    x1, y1, x2, y2 = d['bbox']
                    out.append({
                        'bbox': (int(x1*sw), int(y1*sh), int(x2*sw), int(y2*sh)),
                        'conf': d['conf'],
                        'class': d.get('class', -1)
                    })
                return out

            best_p_dets = scale_dets(best_p_dets_low, scale_w, scale_h)
            best_f_dets = scale_dets(best_f_dets_low, scale_w, scale_h)

            # 6) Create final square thumbnail
            thumbnail = self.create_square_thumbnail(
                best_highres_frame, best_p_dets, best_f_dets)

            # 7) Encode + upload to S3
            _, buffer = cv2.imencode('.jpg', thumbnail)
            thumbnail_bytes = io.BytesIO(buffer.tobytes())

            orig_folder_name = int(Path(row.original_path).stem)
            folder_name = orig_folder_name - (orig_folder_name % 1000)
            media_path = f"images/{folder_name}/{orig_folder_name}.jpg"
            self.store.upload(self.library_name, media_path, thumbnail_bytes)

            # 8) Create the screenshot record
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

        return df
