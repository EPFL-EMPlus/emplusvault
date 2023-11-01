import os
import argparse
import logging
import base64
import io
import re
import numpy as np
import pandas as pd
import cv2
import PIL
import torch
import orjson
import matplotlib.pyplot as plt

import openpifpaf
import openpifpaf.predict

import emv.utils

from enum import Enum
from pathlib import Path
from openpifpaf import transforms
from typing import List, Dict, Tuple, Union, Optional

from emv.utils import FileVideoStream, timeit, dataframe_from_hdf5
from emv.client.get_content import get_frame

from emv.settings import DRIVE_PATH

LOG = emv.utils.get_logger()


KEYPOINTS_NAMES = openpifpaf.Annotation.from_cif_meta(
    openpifpaf.plugins.coco.CocoKp().head_metas[0]).keypoints

for k in ["left_ear", "right_ear", "left_eye", "right_eye"]:
    KEYPOINTS_NAMES.remove(k)

CONNECTIONS = [
    ("nose", "right_shoulder"),
    ("nose", "left_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip")
]

ANGLES_ASSOCIATIONS = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "neck": ("left_shoulder", "nose", "right_shoulder")
}

def keypoint_name_to_id(keypoint):
    return KEYPOINTS_NAMES.index(keypoint)

def format_keypoints_to_read(keypoints):
    return {k:v for k,v in zip(KEYPOINTS_NAMES, keypoints)}


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
PIFPAF_FACTORY = PifPafFactory()


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


@timeit
def process_video(model_name: str, video_bytes: bytes, skip_frame: int = 0, options=None):
    """Process and run pose detection algorithm
       Returns poses jsonl
    """
    def image_reader(frame):
        if frame is None:
            return None
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = PIL.Image.fromarray(image)
        return image_pil

    force = False
    if options and 'force' in options:
        force = options['force']

    # Create new model because it is executed async in another process
    # not really thread-safe :/
    processor = PIFPAF_FACTORY.build_model(model_name)
    # TODO: switch GPU if possible here
    processor.reset()

    # loop over frames from the video file stream
    fvs = FileVideoStream(video_bytes, image_reader)
    fvs.start()
    frame_i = -1

    processed = []
    while fvs.more():
        image_pil = fvs.read()
        if image_pil is None:  # last frame
            break

        frame_i += 1
        if skip_frame > 0 and frame_i % skip_frame != 0:
            continue

        results = processor.process_pil_image(image_pil)
        processed.append({
            'frame': frame_i,
            'data': results
        })
    fvs.stop()

    return processed


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
    Drop keypoints not used: "left_ear", "right_ear", "left_eye", "right_eye".

    Parameters:
    - keypoints (List[float]): The list of keypoints as floats.

    Returns:
    - List[Tuple[float, float, float]]: The reshaped keypoints.
    """
    keypoints = [(keypoints[i], keypoints[i+1], keypoints[i+2]) for i in range(0, len(keypoints), 3)]
    keypoints = keypoints[:1] + keypoints[5:] # Drop unused keypoints
    return keypoints


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
    - float: The computed angle confidence scores (as the mean score of the 3 keypoints)
    """
    vec1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    vec2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    # Check for zero magnitudes or low confidence
    if mag1 == 0 or mag2 == 0:
        return 0.0, 0.0  # Angle is undefined, return 0
    
    if p1[2] < min_confidence or p2[2] < min_confidence or p3[2] < min_confidence:
        return 0.0, 0.0  # Low confidence, return 0
    
    dot_product = np.dot(vec1, vec2)
    cos_theta = dot_product / (mag1 * mag2)

    # Clip value to be in the range [-1, 1] to avoid invalid values due to numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    if angle_deg < 1:
        angle_deg = 0.0

    angle_score = np.mean([p1[2], p2[2], p3[2]])

    return angle_deg, angle_score


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
    associations = [[keypoint_name_to_id(k) for k in assoc] for angle,assoc in ANGLES_ASSOCIATIONS.items()]

    angles = []
    angles_scores = []
    for p1_i, p2_i, p3_i in associations:
        # Check that all required keypoints exist
        if p1_i >= len(keypoints) or p2_i >= len(keypoints) or p3_i >= len(keypoints):
            angles.append(None)
            angles_scores.append(0.0)
            continue

        p1, p2, p3 = keypoints[p1_i], keypoints[p2_i], keypoints[p3_i]

        # Check for incomplete keypoints
        if p1 is None or p2 is None or p3 is None:
            angles.append(None)
            angles_scores.append(0.0)
            continue

        # Check for confidence threshold
        if p1[2] < min_confidence or p2[2] < min_confidence or p3[2] < min_confidence:
            angles.append(None)
            angles_scores.append(0.0)
            continue

        # Compute and store the angle
        angle, angle_score = compute_angle(p1, p2, p3)
        angles.append(angle)
        angles_scores.append(angle_score)

    return angles, angles_scores


def draw_pose(pose, ax = None, cut: bool=True, threshold: float=0.1):
    """
    Draw extracted skeleton on frame.

    Parameters:
    - pose (Dict[str, Any]): The pose data.
    - ax (matplotlib.axes.Axes): The axes to draw on.
    - cut (bool): Whether to cut the frame to the bounding box.
    - threshold (float): The confidence threshold for keypoints. Keypoints with confidence below this value will not be drawn.

    Returns:
    - matplotlib.axes.Axes: The axes with the drawn pose.
    """

    keypoints = pose["keypoints"]
    frame = get_frame(pose["media_id"], pose["frame_number"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(frame)
    ax.scatter([k[0] for k in keypoints if k[2] > threshold], 
               [k[1] for k in keypoints if k[2] > threshold], 
               s=10)
    for c in CONNECTIONS:
        k1 = keypoints[KEYPOINTS_NAMES.index(c[0])]
        k2 = keypoints[KEYPOINTS_NAMES.index(c[1])]
        if k1[2] > threshold and k2[2] > threshold:
            ax.plot([k1[0], k2[0]], 
                    [k1[1], k2[1]], 
                    linewidth=1, color='black')
        
    # cut frame to bbox
    if cut:
        bbox = pose["bbox"]
        ax.set_xlim(int(bbox[0]),int(bbox[0] + bbox[2]))
        ax.set_ylim(int(bbox[1] + bbox[3]), int(bbox[1]))

    ax.axis("off")
    ax.set_aspect('equal')
    plt.tight_layout()

    return ax


standing_still_angle = {
    "left_elbow": 0.9,
    "right_elbow": 0.9,
    "left_shoulder": 0.15,
    "right_shoulder": 0.15,
    "left_hip": 0.95,
    "right_hip": 0.95,
    "left_knee":0.95,
    "right_knee":0.95,
    "neck":0.5
}

sitting_pose_angle = {
    "left_elbow": 0.6,
    "right_elbow": 0.6,
    "left_shoulder": 0.15,
    "right_shoulder": 0.15,
    "left_hip": 0.5,
    "right_hip": 0.5,
    "left_knee":0.5,
    "right_knee":0.5,
    "neck":0.5
}

def drop_pose(input_pose, filter_pose_angles, drop_threshold = 0.1):
    """
    Drop pose unless at least one angle is different from the filter pose angles
    """
    input_angle = input_pose["angle_vec"]
    input_angle_scores = input_pose["angle_score"]
    angles_diff = [np.abs(input_angle[i] - filter_pose_angles[k]) for i,k in enumerate(filter_pose_angles.keys()) if input_angle_scores[i] > 0.1]
    return sum([a > drop_threshold for a in angles_diff]) < 1


def process_frame_data(frame_data: dict, 
                       min_confidence: float = 0.5, 
                       min_valid_keypoints: int = 10, 
                       min_valid_angles: int = 5 ):
    frame_width, frame_height = frame_data['data']['width_height']
    frame_number = frame_data['frame']
    media_id = frame_data['media_id']

    d = {
        'media_id': media_id,
        'frame_number': frame_number,
        'angles': [],
        'angle_vec': [],  # To store normalized angle vectors
        'angle_scores': [], # To store angle confidence scores
        'keypoints': [],
        'bbox': [],
        'frame_width': frame_width,
        'frame_height': frame_height,
        'num_subjects': 0  # Initialize to 0; will increment for each valid person
    }
    
    annotations = frame_data['data']['annotations']
    if not annotations:
        return d
    
    for person in annotations:
        keypoints = person['keypoints']
        bbox = person['bbox']
        reshaped_keypoints = reshape_keypoints(keypoints)
        
        # Count valid keypoints
        valid_keypoints = sum(1 for x, y, c in reshaped_keypoints if c >= min_confidence)
        
        if valid_keypoints < min_valid_keypoints:
            continue  # Skip this person
        
        angles_adjusted, angle_scores = compute_human_angles(reshaped_keypoints, min_confidence)

        # Count valid angles
        valid_angles = sum(1 for angle in angles_adjusted if angle != 0.0)
        
        if valid_angles < min_valid_angles:
            continue  # Skip this person
        
        normalized_angles = normalize_angles(angles_adjusted)
        
        d['angles'].append(angles_adjusted)
        d['angle_vec'].append(normalized_angles.tolist())  # Store normalized angle vector
        d['angle_scores'].append(angle_scores)
        d['keypoints'].append(reshaped_keypoints)
        d['bbox'].append(bbox)
        d['num_subjects'] += 1  # Increment for each valid person

    return d

def extract_frame_data(jsonl_file_path: Union[str, Path], 
                       min_confidence: float = 0.5, 
                       min_valid_keypoints: int = 10, 
                       min_valid_angles: int = 5) -> Dict[int, Dict[str, Union[List[Dict[str, float]], List[List[float]], Dict[str, int]]]]:
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
            
            d = process_frame_data(obj, min_confidence, min_valid_keypoints, min_valid_angles)
            if d['num_subjects'] > 0:
                frame_data[frame_number] = d

    return frame_data

def load_all_poses(poses_folder: Union[str, Path], 
                   drop_poses: List[object] = [standing_still_angle, sitting_pose_angle],
                   drop_threshold: float = 0.1):
    poses_jl = [poses_folder + f for f in os.listdir(poses_folder)]
    poses = []
    for pose_json in poses_jl:
        extracted_data = extract_frame_data(pose_json, 0.5)
        if len(extracted_data.keys()) > 0:
            pose_exp = [[{"frame_number":k, "angle_vec":angle, "angle_score":score, "keypoints":keypoint, "bbox":bbox} 
                         for angle,score,keypoint,bbox 
                         in zip(v["angle_vec"], v["angle_scores"], v["keypoints"], v["bbox"])] for k,v in extracted_data.items()]
            pose_exp = [item for sublist in pose_exp for item in sublist]
            [p.update({"media_id":pose_json.split("/")[-1].split(".")[0]}) for p in pose_exp]
            poses.extend(pose_exp)
    
    for filter_pose in drop_poses:
        poses = [p for p in poses if not drop_pose(p, filter_pose_angles=filter_pose, drop_threshold=drop_threshold)]

    print("Extracted %d poses" %len(poses))

    return poses

def process_all_poses(results: list,
                      drop_poses: List[object] = [standing_still_angle, sitting_pose_angle],
                      drop_threshold: float = 0.1) -> pd.DataFrame:

    poses = [(p["data"]["frames"], p["media_id"]) for p in results]
    [[p.update({"media_id":data[1]}) for p in data[0]][0] for data in poses]
    poses = [p[0] for p in poses]
    poses = [process_frame_data(pose) for poses_list in poses for pose in poses_list]

    poses_flat = []
    for pose in poses:
        pose_exp = [{
                        "media_id":pose["media_id"], 
                        "frame_number":pose["frame_number"], 
                        "angle_vec":angle, 
                        "angle_score":score, 
                        "keypoints":keypoint, 
                        "bbox":bbox
                    } 
                    for angle, score, keypoint, bbox 
                    in zip(pose["angle_vec"], pose["angle_scores"], pose["keypoints"], pose["bbox"])]
        poses_flat.extend(pose_exp)

    if np.sum([len(pose["angle_vec"]) for pose in poses]) == len(poses_flat):
        print(f"All {len(poses_flat)} poses were flattened correctly")

    # Drop uninteresting poses
    for filter_pose in drop_poses:
        poses_flat = [p for p in poses_flat 
        if not drop_pose(p, filter_pose_angles=filter_pose, drop_threshold=drop_threshold)]
    if len(drop_poses) > 0:
        print(f"{len(poses_flat)} poses left after filtering")

    # Merge with metadata
    data = dataframe_from_hdf5(DRIVE_PATH, "metadata")
    data["seq_id"] = data.seq_id.map(lambda x: f"ioc-{x}")
    pose_df = pd.merge(pd.DataFrame(poses_flat), data[["seq_id", "sport"]], left_on="media_id", right_on="seq_id")

    return pose_df 
