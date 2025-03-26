import os
import argparse
import logging
import base64
import io
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import cv2
import PIL
import torch
import orjson
import struct
from io import BytesIO

import emv.utils
import math

from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

from emv.utils import FileVideoStream, timeit
from emv.settings import get_secret
from emv.storage.storage import get_storage_client


LOG = emv.utils.get_logger()


class ExtendedEnum(str, Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class PifPafModel(ExtendedEnum):
    fast = "pifpaf_fast"
    accurate = "pifpaf_accurate"
    fast_wholebody = "pifpaf_fast_wholebody"
    wb_hand = "pifpaf_wb_hand"
    accurate_wholebody = "pifpaf_accurate_wholebody"
    pose_track = "pifpaf_posetrack"

    animal = 'pifpaf_animal'
    fast_car = "pifpaf_fast_car"
    accurate_car = "pifpaf_accurate_car"

    # fast_center = "pifpaf_fast_center"
    # accurate_center = "pifpaf_accurate_center"
    hand = "pifpaf_hand"

    nuscenes = "NuScenes_2D"
    cocodet = "cocodet"


class AnnotationCategory(ExtendedEnum):
    human = 1
    hands_only = 100
    fast_car = 3
    accurate_car = 300
    animal = 4


MODEL_MAP = {
    PifPafModel.fast: "shufflenetv2k16",  # shufflenetv2k16-center, shufflenetv2k16
    # shufflenetv2k30-center, shufflenetv2k30
    PifPafModel.accurate: "shufflenetv2k30",
    # shufflenetv2k16-wb-hand or shufflenetv2k16-wholebody
    PifPafModel.fast_wholebody: "shufflenetv2k16-wholebody",
    PifPafModel.wb_hand: "shufflenetv2k16-wb-hand",
    PifPafModel.accurate_wholebody: "shufflenetv2k30-wholebody",
    PifPafModel.pose_track: "tshufflenetv2k30",
    PifPafModel.fast_car: "shufflenetv2k16-apollo-24",
    PifPafModel.accurate_car: "shufflenetv2k16-apollo-66",
    PifPafModel.animal: "shufflenetv2k30-animalpose",
    # PifPafModel.fast_center: "v2-shufflenetv2k16-center",
    # PifPafModel.accurate_center: "v2-shufflenetv2k30-center",
    PifPafModel.hand: "shufflenetv2k16-hand",
    # "shufflenetv2k16-nuscenes",#"resnet18-cocodet",# "shufflenetv2k16-nuscenes", #"resnet18-cocodet", #"shufflenetv2k16-nuscenes",
    PifPafModel.nuscenes: "shufflenetv2k16-nuscenes",
    PifPafModel.cocodet: "resnet18-cocodet",
}


def get_torch_checkpoints_dir():
    base_dir = None
    if hasattr(torch, 'hub') and hasattr(torch.hub, 'get_dir'):
        # new in pytorch 1.6.0
        base_dir = torch.hub.get_dir()
    elif os.getenv('TORCH_HOME'):
        base_dir = os.getenv('TORCH_HOME')
    elif os.getenv('XDG_CACHE_HOME'):
        base_dir = os.path.join(os.getenv('XDG_CACHE_HOME'), 'torch')
    else:
        base_dir = os.path.expanduser(os.path.join('~', '.cache', 'torch'))
    return os.path.join(base_dir, 'checkpoints')


def download_model():
    openpifpaf.network.Factory(
        checkpoint="shufflenetv2k16", download_progress=True).factory()


class StandardProcessor(object):
    def __init__(self, name, args, category_id=None):
        self.name = name
        self.cat_id = category_id
        model_cpu, _ = openpifpaf.network.Factory().factory()

        self.model = model_cpu.to(args.device)
        head_metas = [hn.meta for hn in self.model.head_nets]
        self.processor = openpifpaf.decoder.factory(head_metas)
        self.preprocess = self.preprocess_factory(args)
        self.device = args.device
        self.args = args

    def preprocess_factory(self, args):
        rescale_t = None
        if 'long_edge' in args and args.long_edge:
            rescale_t = transforms.RescaleAbsolute(
                args.long_edge, fast=args.fast_rescaling)

        pad_t = None
        if args.batch_size > 1:
            assert args.long_edge, '--long-edge must be provided for batch size > 1'
            pad_t = transforms.CenterPad(args.long_edge)
        else:
            pad_t = transforms.CenterPadTight(16)

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            rescale_t,
            pad_t,
            transforms.EVAL_TRANSFORM,
        ])

    def update_parameter(self, arg_name, arg_value):
        if arg_name in self.args:
            self.args.__dict__[arg_name] = arg_value
            return arg_name
        return None

    def from_base64(self, b64image):
        if not b64image:
            return None

        # return {
        #     'annotations': [],
        #     'width_height': (640, 480),
        #     'model_name': "test"
        # }

        try:
            imgstr = re.search(r'base64,(.*)', b64image).group(1)
        except AttributeError as e:
            LOG.error(e)
            return None
        image_bytes = io.BytesIO(base64.b64decode(imgstr))
        return self.from_image_bytes(image_bytes)

    def from_image_bytes(self, image_bytes):
        try:
            im = PIL.Image.open(image_bytes).convert('RGB')
        except PIL.UnidentifiedImageError as e:
            LOG.error(e)
            return None
        return self.process_pil_image(im)

    def process_pil_image(self, im):
        # start = time.time()
        processed_image, _, meta = self.preprocess(im, [], None)

        image_tensors_batch = torch.unsqueeze(processed_image.float(), 0)
        pred_anns = self.processor.batch(
            self.model, image_tensors_batch, device=self.device)[0]
        # All predictions in the image, rescaled to the original image size

        res = []
        for ann in pred_anns:
            inv = ann.inverse_transform(meta)
            if self.cat_id:
                inv.category_id = int(self.cat_id)
            res.append(inv.json_data())

        # LOG.debug(f'processing time: {time.time() - start}')
        return {
            'annotations': res,
            'width_height': (int(meta['width_height'][0]), int(meta['width_height'][1])),
            'model_name': self.name
        }

    def reset(self):
        for d in self.processor.decoders:
            # print('Available decoders', self.processor.decoders)
            if isinstance(d, openpifpaf.decoder.tracking_pose.TrackingPose):
                # print('frame number before: ', d.frame_number)
                d.reset()
                # print('frame number after:', d.frame_number)


class PifPafFactory(object):
    def __init__(self):
        self.has_init = False
        self.args = None
        parser = argparse.ArgumentParser(
            prog="pifpaf",
            description=__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False
        )

        openpifpaf.decoder.cli(parser)
        openpifpaf.network.Factory.cli(parser)

        openpifpaf.logger.cli(parser)
        self.args, _ = parser.parse_known_args()

        # print(self.args)
        self.args.disable_cuda = False
        # self.args.long_edge = 321 # 161
        self.args.batch_size = 0
        self.args.fast_rescaling = True
        # self.args.base_name = False
        self.args.quiet = True
        openpifpaf.logger.configure(self.args, logging.getLogger('openpifpaf'))

        self.args.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.args.device = torch.device('cuda')
        LOG.debug(f'neural network device: {self.args.device}')

    def configure(self):
        openpifpaf.decoder.configure(self.args)
        openpifpaf.network.Factory.configure(self.args)
        # openpifpaf.show.configure(self.args)
        # openpifpaf.visualizer.configure(self.args)

    def build_fast_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.fast]
        self.args.decoder = ['cifcaf:0']
        self.configure()
        return StandardProcessor(PifPafModel.fast, self.args, AnnotationCategory.human)

    def build_accurate_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.accurate]
        self.args.force_complete_pose = True
        self.args.decoder = ['cifcaf:0']
        self.configure()
        return StandardProcessor(PifPafModel.accurate, self.args, AnnotationCategory.human)

    def build_fast_wholebody_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.fast_wholebody]
        self.args.decoder = ['cifcaf:0']
        self.configure()
        return StandardProcessor(PifPafModel.fast_wholebody, self.args, AnnotationCategory.human)

    def build_fast_wb_hands_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.wb_hand]
        self.args.decoder = ['cifcaf:0']
        self.configure()
        return StandardProcessor(PifPafModel.wb_hand, self.args, AnnotationCategory.human)

    def build_accurate_wholebody_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.accurate_wholebody]
        self.args.decoder = ['cifcaf:0']
        self.configure()
        return StandardProcessor(PifPafModel.accurate_wholebody, self.args, AnnotationCategory.human)

    def build_posetrack_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.pose_track]
        # self.args.long_edge = 321
        # self.args.fast_rescaling = True
        self.args.decoder = ['trackingpose:0']
        self.configure()
        proc = StandardProcessor(
            PifPafModel.pose_track, self.args, AnnotationCategory.human)
        proc.reset()
        return proc

    def build_hand_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.hand]
        self.args.decoder = ['cifcaf:0']  # cifcaf:1 can work too
        self.configure()
        return StandardProcessor(PifPafModel.hand, self.args, AnnotationCategory.hands_only)

    def build_animal_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.animal]
        self.args.decoder = ['cifcaf:0']
        self.configure()
        return StandardProcessor(PifPafModel.animal, self.args, AnnotationCategory.animal)

    def build_fast_car_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.fast_car]
        self.args.decoder = ['cifcaf:0']
        self.configure()
        return StandardProcessor(PifPafModel.fast_car, self.args, AnnotationCategory.fast_car)

    def build_accurate_car_pifpaf(self):
        self.args.checkpoint = MODEL_MAP[PifPafModel.accurate_car]
        self.args.decoder = ['cifcaf:0']
        self.configure()
        return StandardProcessor(PifPafModel.accurate_car, self.args, AnnotationCategory.accurate_car)

    def build_nuscenes(self):
        self.args.checkpoint = "shufflenetv2k16-nuscenes"
        self.args.decoder = ['cifdet:0']
        self.configure()
        return StandardProcessor(PifPafModel.nuscenes, self.args)

    def build_cocodet(self):
        self.args.checkpoint = "resnet18-cocodet"
        self.args.decoder = ['cifdet:0']
        self.configure()
        return StandardProcessor(PifPafModel.cocodet, self.args)

    def build_model(self, model_name):
        if model_name == PifPafModel.fast:
            return self.build_fast_pifpaf()
        elif model_name == PifPafModel.accurate:
            return self.build_accurate_pifpaf()
        elif model_name == PifPafModel.fast_wholebody:
            return self.build_fast_wholebody_pifpaf()
        elif model_name == PifPafModel.wb_hand:
            return self.build_fast_wb_hands_pifpaf()
        elif model_name == PifPafModel.accurate_wholebody:
            return self.build_accurate_wholebody_pifpaf()
        elif model_name == PifPafModel.pose_track:
            return self.build_posetrack_pifpaf()
        elif model_name == PifPafModel.accurate_car:
            return self.build_accurate_car_pifpaf()
        elif model_name == PifPafModel.animal:
            return self.build_animal_pifpaf()
        elif model_name == PifPafModel.nuscenes:
            return self.build_nuscenes()
        elif model_name == PifPafModel.cocodet:
            return self.build_cocodet()
        else:
            LOG.error(f'Model: {model_name} not implemented')
            return None


_model_cache = dict()
# PIFPAF_FACTORY = PifPafFactory()


def get_model(model_name):
    if model_name in _model_cache:
        LOG.debug(f"Using cached model {model_name}")
        return _model_cache[model_name]
    else:
        LOG.debug(f"Building model{model_name}")
        mod = PIFPAF_FACTORY.build_model(model_name)
        if not mod:
            return None
        _model_cache[model_name] = mod
        return _model_cache[model_name]


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_size = int(Path(video_path).stat().st_size)

    return {
        'width': w,
        'height': h,
        'fps': fps,
        'total_frames': n_frames,
        'video_size': video_size,
        'video_path': str(video_path)
    }


def get_tmp_lock_path(annot_path: Path, video_path: Path) -> Path:
    # Check temporary file while the video is processing to prevent
    # invalid results
    tmp_file = video_path.stem + '.tmp'
    tmp_file = annot_path.parent.joinpath(tmp_file)
    return tmp_file.resolve()


def get_annotations_path(model_name: str, video_path: Path, data_folder: Path) -> Path:
    data_path = model_name + '_'
    if model_name.startswith('pifpaf'):
        data_path = data_path + get_pifpaf_version()
    else:
        LOG.error(f'Model not implemented {model_name}')
        return None

    data_path = data_folder.joinpath(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    output_file = video_path.stem + '.jl'
    annot_path = data_path.joinpath(output_file)
    return annot_path


def check_if_valid_annotations(annot_path: Path) -> bool:
    """Check if already processed and without error"""
    if annot_path.exists():
        return True
    return False


def get_keypoints_from_image(image_pil) -> Dict:
    PIFPAF_FACTORY = PifPafFactory()
    processor = PIFPAF_FACTORY.build_model(PifPafModel.fast)
    processor.reset()

    results = processor.process_pil_image(image_pil)
    return results


def extract_pose_frames(data: List[dict]) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Extracts pose keypoints from the input data along with the frame number, excluding unnecessary keypoints.
    """

    EXCLUDED_KEYPOINTS = ()  #  (1, 2, 3, 4)

    pose_frames = []
    for frame_data in data:
        frame_number = frame_data['frame']
        frame_annotations = frame_data['data']['annotations']
        frame_poses = []

        for annotation in frame_annotations:
            keypoints = annotation['keypoints']
            person_id = annotation['track_id']
            pose = [
                (keypoints[i], keypoints[i + 1])
                for j, i in enumerate(range(0, len(keypoints), 3))
                if j not in EXCLUDED_KEYPOINTS
            ]
            # if nose is not visible, take the average of the eyes or ears instead if they are available
            if pose[0][0] == 0 and pose[0][1] == 0:
                if pose[1][0] != 0 and pose[1][1] != 0 and pose[2][0] != 0 and pose[2][1] != 0:
                    pose[0] = ((pose[1][0] + pose[2][0]) / 2,
                               (pose[1][1] + pose[2][1]) / 2)
                elif pose[3][0] != 0 and pose[3][1] != 0 and pose[4][0] != 0 and pose[4][1] != 0:
                    pose[0] = ((pose[3][0] + pose[4][0]) / 2,
                               (pose[3][1] + pose[4][1]) / 2)

            frame_poses.append((frame_number, person_id, pose))

        pose_frames.extend(frame_poses)

    return pose_frames


def write_pose_to_binary_file(feature_id: int, no_frames: int, skeleton_bones_count: int, video_resolution: Tuple[int, int], pose_frames: List[Tuple[int, List[Tuple[float, float]]]], bucket_name: str, object_name: str) -> None:
    """
    Writes the pose from a video to a binary file.

    :param pose_frames: A list of frames, where each frame corresponds to a video frame, and each frame is a list of its joints positions
    :param filename: The name of the file to write the pose to

    The binary data structure is as follows:

    Header :
        Feature identifier (uint32)
        Total frames in binary file, a multiple of frames in video due to multiple persons in a frame (uint32)
        Total frames in the video (uint32)
        Skeleton bones count (uint8)
        Video resolution width and height (2 * uint16)

    Data :
        for each frame :
            for each bone in pose :
                frame number (uint32) + location 2D vector (2 * uint16) -> in video frame coordinates referential (i.e. in range [0, 720] * [0, 1024])

    """

    binary_stream = BytesIO()

    # Write header
    # Feature identifier (uint32)
    binary_stream.write(struct.pack('<I', feature_id))
    binary_stream.write(struct.pack('<I', len(pose_frames)))
    binary_stream.write(struct.pack('<I', no_frames))
    # Skeleton bones count (uint8)
    binary_stream.write(struct.pack('<B', skeleton_bones_count))
    # Video resolution (2 * uint16)
    binary_stream.write(struct.pack(
        '<HH', video_resolution[0], video_resolution[1]))

    # Write pose data
    for frame_no, person_id, pose in pose_frames:
        assert len(
            pose) == skeleton_bones_count, f"Frame {frame_no} has incorrect bone count!"

        binary_stream.write(struct.pack('<I', frame_no))  # uint32
        binary_stream.write(struct.pack('<I', person_id))  # uint32

        for joint_x, joint_y in pose:
            coord_x = min(max(0, int(joint_x)), video_resolution[0] - 1)
            coord_y = min(max(0, int(joint_y)), video_resolution[1] - 1)
            binary_stream.write(struct.pack(
                '<HH', coord_x, coord_y))  # 2 * uint16

    # Upload to storage
    binary_stream.seek(0)
    storage_client = get_storage_client()
    success = storage_client.upload(
        bucket_name, object_name, binary_stream)
    if success:
        print(f"File successfully uploaded to {bucket_name}/{object_name}")
    else:
        print("File upload failed.")


def read_pose_from_binary_file(binary_data: BytesIO) -> Tuple[int, int, Tuple[int, int], List[Tuple[int, List[Tuple[int, int]]]]]:
    """
    Reads pose data from a binary file stored in memory. This function is mostly used for testing purposes.
    """
    binary_data.seek(0)
    feature_id = struct.unpack('<I', binary_data.read(4))[0]
    frame_count = struct.unpack('<I', binary_data.read(4))[0]
    skeleton_bones_count = struct.unpack('<B', binary_data.read(1))[0]
    video_resolution = struct.unpack('<HH', binary_data.read(4))

    pose_frames = []
    for _ in range(frame_count):
        frame_number = len(pose_frames)
        pose = [
            struct.unpack('<HH', binary_data.read(4))
            for _ in range(skeleton_bones_count)
        ]
        pose_frames.append((frame_number, pose))

    return feature_id, skeleton_bones_count, video_resolution, pose_frames


def process_video(
    model_name: str,
    video_bytes: bytes,
    skip_frame: int = 0,
    options: dict = None,
    detection_conf_threshold: float = 0.3,
    keypoint_conf_threshold: float = 0.5,
    matching_distance_threshold: float = 100.0,
    max_missing_frames: int = 10,
    top_n_detections: int = 10
):
    """
    Processes video bytes to detect and track poses using a YOLO Pose model and Kalman filters.
    Returns:
        (processed, images) where:
          - processed: List[{'frame': int, 'data': [ { 'track_id':..., 'bbox':..., 'pose':...} ]}]
          - images: List of the original frames in JPEG bytes
    """
    import cv2
    import tempfile
    import numpy as np
    import io
    import os
    from PIL import Image
    from ultralytics import YOLO

    def create_kalman_filter(initial_center):
        """
        Create and initialize a Kalman filter for tracking.
        """
        kf = cv2.KalmanFilter(4, 2)
        # State vector: [x, y, dx, dy]
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        # Measurement vector: [x, y]
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array(
            [[initial_center[0]], [initial_center[1]], [0], [0]], np.float32)
        return kf

    #  Face keypoints not needed for pose matching
    EXCLUDED_KEYPOINTS = (1, 2, 3, 4)

    # Write the input video bytes to a temporary file
    input_temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        input_temp.write(video_bytes)
        input_temp.flush()
        input_path = input_temp.name
    finally:
        input_temp.close()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        os.remove(input_path)
        raise IOError("Could not open video from bytes")

    model = YOLO(model_name)

    # Tracking state
    # Each track is a dict: {'id', 'box', 'center', 'kf', 'missing_frames', 'color', ...}
    person_tracks = []
    next_id = 0

    frame_count = 0
    processed = []
    images = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # If skipping frames is desired
        if skip_frame > 0 and frame_count % (skip_frame + 1) != 0:
            frame_count += 1
            continue

        results = model.predict(
            frame, conf=detection_conf_threshold, verbose=False)
        current_detections = []

        for result in results:
            if result.keypoints is not None and result.boxes is not None:
                boxes = result.boxes
                # shape (N, 13, 3) if 13 keypoints, for example
                keypoints = result.keypoints.data

                # Compute areas and sort detections by area (largest first)
                areas = []
                for i, box in enumerate(boxes):
                    # boxes[i].xyxy -> shape (1, 4) => x1, y1, x2, y2
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    area = (x2 - x1) * (y2 - y1)
                    areas.append((area, i))

                if not areas:
                    continue

                areas.sort(reverse=True)
                top_indices = [idx for (area, idx) in areas[:top_n_detections]]

                # Save only top N detections
                for idx in top_indices:
                    box = boxes[idx]
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # center of bbox
                    kps = keypoints[idx]  # shape (K, 3) => x, y, conf

                    current_detections.append({
                        'center': (cx, cy),
                        'box': (x1, y1, x2, y2),
                        'kps': kps
                    })

        # -- Kalman Filter Tracking Step --
        # 1) Predict new positions for existing tracks
        for track in person_tracks:
            pred = track['kf'].predict()
            track['predicted_center'] = (pred[0][0], pred[1][0])

        # 2) Match current detections to existing tracks by distance
        assigned_tracks = set()
        assigned_detections = set()
        for i, track in enumerate(person_tracks):
            if track['missing_frames'] > max_missing_frames:
                continue
            best_match = None
            best_distance = float('inf')
            best_det_idx = -1

            tx, ty = track['predicted_center']
            for j, detection in enumerate(current_detections):
                if j in assigned_detections:
                    continue
                cx, cy = detection['center']
                distance = np.hypot(tx - cx, ty - cy)
                if distance < matching_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = detection
                    best_det_idx = j

            if best_match is not None:
                # Update track using the detection measurement
                measurement = np.array([[np.float32(best_match['center'][0])],
                                        [np.float32(best_match['center'][1])]])
                track['kf'].correct(measurement)
                state = track['kf'].statePost
                track['center'] = (state[0][0], state[1][0])
                track['box'] = best_match['box']
                track['missing_frames'] = 0
                best_match['track_id'] = track['id']
                assigned_tracks.add(i)
                assigned_detections.add(best_det_idx)
            else:
                track['missing_frames'] += 1

        # 3) Create new tracks for unmatched detections
        for j, detection in enumerate(current_detections):
            if j not in assigned_detections:
                track_id = next_id
                next_id += 1
                kf = create_kalman_filter(detection['center'])
                new_track = {
                    'id': track_id,
                    'box': detection['box'],
                    'center': detection['center'],
                    'kf': kf,
                    'missing_frames': 0
                }
                detection['track_id'] = track_id
                person_tracks.append(new_track)

        # 4) Remove tracks that have been missing for too long
        person_tracks = [
            t for t in person_tracks if t['missing_frames'] <= max_missing_frames]

        frame_data = {'annotations': [],
                      'width_height': frame.shape[:2], 'model_name': model_name}
        img_h, img_w = frame.shape[:2]
        for detection in current_detections:
            if 'track_id' in detection:
                track_id = detection['track_id']
                (x1, y1, x2, y2) = detection['box']
                # Ensure bounding box is within image boundaries
                x1_clamped = max(0, min(x1, img_w))
                x2_clamped = max(0, min(x2, img_w))
                y1_clamped = max(0, min(y1, img_h))
                y2_clamped = max(0, min(y2, img_h))

                # Convert keypoints to the same structure as your old code
                # YOLO keypoints are (x, y, confidence)
                kps = detection['kps']
                pose_data = []
                for kp_idx, (kp_x, kp_y, kp_conf) in enumerate(kps.tolist()):

                    if kp_idx in EXCLUDED_KEYPOINTS:
                        continue

                    # clamp keypoints as well
                    kp_x_clamped = max(0, min(kp_x, img_w))
                    kp_y_clamped = max(0, min(kp_y, img_h))

                    pose_data.extend([
                        kp_x_clamped, kp_y_clamped, float(
                            kp_conf),  # 'confidence'
                    ])

                frame_data['annotations'].append({
                    'track_id': track_id,
                    'bbox': [x1_clamped, y1_clamped, x2_clamped, y2_clamped],
                    'keypoints': pose_data,
                })

        img_bytes = io.BytesIO()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_img.save(img_bytes, format="JPEG")

        images.append(img_bytes.getvalue())
        processed.append({'frame': frame_count, 'data': frame_data})

        frame_count += 1

    cap.release()
    os.remove(input_path)

    return processed, images


def get_pifpaf_version():
    return openpifpaf.__version__


def get_available_pifpaf_models():
    return list(PifPafModel)


def get_gpu_props(gpu_id):
    return torch.cuda.get_device_properties(gpu_id)


def get_current_device_name():
    gpu_id = torch.cuda.current_device()
    return get_gpu_props(gpu_id).name


def reshape_keypoints(keypoints: List[float]) -> List[Tuple[float, float, float]]:
    """
    Reshape a list of keypoints into a list of tuples (x, y, confidence).

    Parameters:
    - keypoints (List[float]): The list of keypoints as floats.

    Returns:
    - List[Tuple[float, float, float]]: The reshaped keypoints.
    """
    return [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]


def compute_angle(p1: Tuple[float, float, float],
                  p2: Tuple[float, float, float],
                  p3: Tuple[float, float, float],
                  min_confidence: float = 0.5) -> float:
    """
    Compute the angle at p2 given points p1, p2, and p3 in 2D space, using a confidence threshold for filtering.

    Parameters:
    - p1, p2, p3 (Tuple[float, float, float]): Points with x, y coordinates and confidence level.
    - min_confidence (float): The minimum confidence level for a keypoint to be considered.

    Returns:
    - float: The computed angle in degrees.
    """
    vec1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    vec2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    # Check for zero magnitudes or low confidence
    if mag1 == 0 or mag2 == 0:
        return 0.0  # Angle is undefined, return 0

    if p1[2] < min_confidence or p2[2] < min_confidence or p3[2] < min_confidence:
        return 0.0  # Low confidence, return 0

    dot_product = np.dot(vec1, vec2)
    cos_theta = dot_product / (mag1 * mag2)

    # Clip value to be in the range [-1, 1] to avoid invalid values due to numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    if angle_deg < 1:
        angle_deg = 0.0

    return angle_deg


def normalize_angles(angles: List[Optional[float]]) -> np.ndarray:
    """
    Normalize angles to the range [0, 1].

    Parameters:
    - angles (List[Optional[float]]): List of angles in degrees.

    Returns:
    - np.ndarray: NumPy array of normalized angles.
    """
    # Replace None values with 0 and convert to NumPy array
    angles_clean = np.array(
        [angle if angle is not None else 0 for angle in angles])

    # Normalize angles to [0, 1]
    normalized_angles = angles_clean / 180.0

    return normalized_angles


def compute_human_angles(keypoints: List[Tuple[float, float, float]], min_confidence: float = 0.5) -> List[Optional[float]]:
    """
    Compute meaningful angles for human pose based on keypoints.

    Parameters:
    - keypoints (List[Tuple[float, float, float]]): List of keypoints (x, y, confidence).
    - min_confidence (float): Minimum confidence level to consider a keypoint valid.

    Returns:
    - List[Optional[float]]: List of computed angles in degrees. Missing or undefined angles are set to None.
    """
    # Define the associations to compute angles
    associations = [
        (5, 7, 9),   # Left elbow
        (6, 8, 10),  # Right elbow
        (11, 13, 15),  # Left knee
        (12, 14, 16),  # Right knee
        (5, 11, 13),  # Left hip
        (6, 12, 14),  # Right hip
        (3, 5, 7),    # Left shoulder
        (4, 6, 8),    # Right shoulder
        (5, 0, 6),    # Neck
        (13, 15, 16),  # Left ankle
        (14, 16, 15)  # Right ankle
    ]

    angles = []
    for p1_i, p2_i, p3_i in associations:
        # Check that all required keypoints exist
        if p1_i >= len(keypoints) or p2_i >= len(keypoints) or p3_i >= len(keypoints):
            angles.append(None)
            continue

        p1, p2, p3 = keypoints[p1_i], keypoints[p2_i], keypoints[p3_i]

        # Check for incomplete keypoints
        if p1 is None or p2 is None or p3 is None:
            angles.append(None)
            continue

        # Check for confidence threshold
        if p1[2] < min_confidence or p2[2] < min_confidence or p3[2] < min_confidence:
            angles.append(None)
            continue

        # Compute and store the angle
        angle = compute_angle(p1, p2, p3)
        angles.append(angle)

    return angles


def extract_frame_data(jsonl_file_path: Union[str, Path], min_confidence: float = 0.5, min_valid_keypoints: int = 10, min_valid_angles: int = 5) -> Dict[int, Dict[str, Union[List[Dict[str, float]], List[List[float]], Dict[str, int]]]]:
    """
    Extract and compute angles, keypoints, bounding boxes, and frame dimensions for each frame from a JSONLines file.

    Parameters:
    - jsonl_file_path (str): The path to the JSONLines file.
    - min_valid_keypoints (int): Minimum number of valid keypoints required to include a person.
    - min_valid_angles (int): Minimum number of valid angles required to include a person.

    Returns:
    - Dict[int, Dict[str, Union[List[Dict[str, float]], List[List[float]], Dict[str, int]]]]: The extracted and computed data.
    """
    frame_data = {}

    with open(str(jsonl_file_path), 'rb') as f:
        for line in f:
            obj = orjson.loads(line)
            frame_number = obj['frame']
            annotations = obj['data']['annotations']

            if not annotations:
                continue
            frame_width, frame_height = obj['data']['width_height']

            d = {
                'angles': [],
                'angle_vec': [],  # To store normalized angle vectors
                'keypoints': [],
                'bbox': [],
                'frame_width': frame_width,
                'frame_height': frame_height,
                'num_subjects': 0  # Initialize to 0; will increment for each valid person
            }

            for person in annotations:
                keypoints = person['keypoints']
                bbox = person['bbox']
                reshaped_keypoints = reshape_keypoints(keypoints)

                # Count valid keypoints
                valid_keypoints = sum(
                    1 for x, y, c in reshaped_keypoints if c >= min_confidence)

                if valid_keypoints < min_valid_keypoints:
                    continue  # Skip this person

                angles_adjusted = compute_human_angles(
                    reshaped_keypoints, min_confidence)

                # Count valid angles
                valid_angles = sum(
                    1 for angle in angles_adjusted if angle != 0.0)

                if valid_angles < min_valid_angles:
                    continue  # Skip this person

                normalized_angles = normalize_angles(angles_adjusted)

                d['angles'].append(angles_adjusted)
                # Store normalized angle vector
                d['angle_vec'].append(normalized_angles.tolist())
                d['keypoints'].append(reshaped_keypoints)
                d['bbox'].append(bbox)
                d['num_subjects'] += 1  # Increment for each valid person

            if d['num_subjects'] > 0:
                frame_data[frame_number] = d

    return frame_data


def get_angle_feature_vector(keypoints: List[List[float]]):
    def calculate_angle(points):
        assert len(
            points) == 3, "Three points are required to calculate the angles"

        hip1, hip2, ref = np.array(points)

        if (hip1[0] == ref[0] and hip1[1] == ref[1]) or \
                (hip2[0] == ref[0] and hip2[1] == ref[1]) or \
                (hip1[0] == hip2[0] and hip1[1] == hip2[1]):
            raise ValueError("Points cannot be the same")

        # Calculate the lengths of the sides of the triangle
        a = np.linalg.norm(hip2 - ref)
        b = np.linalg.norm(hip1 - ref)
        c = np.linalg.norm(hip1 - hip2)

        # Law of cosines to find the angles
        angle_hip2_hip1_ref = np.degrees(
            np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))
        angle_hip1_hip2_ref = np.degrees(
            np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
        angle_hip1_ref_hip2 = np.degrees(
            np.arccos((a**2 + b**2 - c**2) / (2 * a * b)))

        return angle_hip2_hip1_ref, angle_hip1_hip2_ref, angle_hip1_ref_hip2

    angles = []

    # Using indices for reference points, these should be the two hips.
    ref_indices = [7, 8]  # Indices of the reference points

    # Calculate angles for each keypoint relative to the two reference points
    for i, keypoint in enumerate(keypoints):
        if i not in ref_indices:
            angles.extend(calculate_angle(
                [keypoints[ref_indices[0]], keypoint, keypoints[ref_indices[1]]]))

    feature_vector = np.array(angles)
    return feature_vector / 180.0


def get_poem_feature_vector(keypoints: List[List[float]]):
    keypoint_names = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                      'right_knee', 'left_ankle', 'right_ankle']

    def create_dataframe(poses):
        data = []
        for pose in poses:
            row = []
            for kp in pose:
                try:
                    row.extend([kp[0], kp[1], kp[2]])
                except IndexError:
                    row.extend([kp[0], kp[1], 0.9])
            data.append(row)

        columns = []
        for kp in keypoint_names:
            kp = kp.replace('nose', 'nose_tip')
            columns.extend([f'image/object/part/{kp.upper()}/center/x',
                           f'image/object/part/{kp.upper()}/center/y', f'image/object/part/{kp.upper()}/score'])

        return pd.DataFrame(data, columns=columns)

    # normalize x and y cords to be between 0 and 1
    def normalize(df):
        for kp in keypoint_names:
            kp = kp.replace('nose', 'nose_tip')
            df[f'image/object/part/{kp.upper()}/center/x'] /= df['image/width']
            df[f'image/object/part/{kp.upper()}/center/y'] /= df['image/height']

        return df

    import subprocess

    df = create_dataframe([keypoints])
    df.insert(0, 'image/width', 1280)
    df.insert(1, 'image/height', 720)
    df = normalize(df)
    df.to_csv(get_secret("POEM_SAVE_LOC"), index=False)
    #  Run the inference script
    subprocess.run(get_secret("POEM_SCRIPT"),
                   shell=True, executable="/bin/bash")
    df_poem = pd.read_csv(get_secret("POEM_RETR_LOC"), header=None)
    return df_poem.values[0]


def filter_poses(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Filter poses based distance to other poses. We want to keep poses that are as unique as possible.
    """
    X = np.array(df.angle_vec.tolist())

    # fill in nan values with the averages of the columns
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

    # create a KDTree
    tree = KDTree(X)
    distances, indices = tree.query(X, k=2)

    to_keep = np.ones(len(X), dtype=bool)
    for i in range(len(X)):
        if not to_keep[i]:
            continue  # Skip points that have already been marked for removal

        # Find neighbors within the threshold
        indices = tree.query_radius([X[i]], r=threshold)[0]

        # Mark the neighbors (except the current point itself) for removal
        for index in indices:
            if index != i:
                to_keep[index] = False

    return df[to_keep]


def centralize_and_scale_keypoints(keypoints, reference_points=['left_hip', 'right_hip'], target_diameter=1):

    keypoint_names = [
        'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]
    keypoints = np.array(keypoints)

    # Calculate the reference point (midpoint between hips)
    left_hip = keypoints[keypoint_names.index(reference_points[0])]
    right_hip = keypoints[keypoint_names.index(reference_points[1])]
    center_point = (left_hip + right_hip) / 2

    # Translate pose to center it at the origin
    translated_keypoints = keypoints - center_point

    # Determine the maximum distance from the center point to any keypoint
    max_distance = np.max(np.linalg.norm(translated_keypoints, axis=1))

    # Calculate the scale factor to make the maximum distance equal to 0.5
    scale_factor = 0.5 / max_distance

    # Scale keypoints to have the target diameter
    scaled_keypoints = translated_keypoints * scale_factor

    return scaled_keypoints
