import os
import numpy as np
import pandas as pd
import ast

import matplotlib.pyplot as plt

import openpifpaf

import emv.utils

from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

from emv.utils import dataframe_from_hdf5
from emv.client.get_content import get_frame

from emv.settings import DRIVE_PATH


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

FILTER_POSES = {
    "standing_still_angle" : {
        "left_elbow": 0.9,
        "right_elbow": 0.9,
        "left_shoulder": 0.15,
        "right_shoulder": 0.15,
        "left_hip": 0.95,
        "right_hip": 0.95,
        "left_knee":0.95,
        "right_knee":0.95,
        "neck":0.5
    },
    "sitting_pose_angle" : {
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
}

def keypoint_name_to_id(keypoint):
    return KEYPOINTS_NAMES.index(keypoint)

def format_keypoints_to_read(keypoints):
    return {k:v for k,v in zip(KEYPOINTS_NAMES, keypoints)}



# PROCESS KEYPOINTS TO ANGLES

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
    keypoints = keypoints[:1] + keypoints[5:] # Drop unused keypoints from the face
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


# DRAW POSES

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

    if frame is None:
        return ax
        
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


# LOAD AND PROCESS POSES DATA

def compare_pose(input_pose, filter_pose_angles, drop_threshold = 0.1):
    """
    Drop pose unless at least one angle is different from the filter pose angles
    """
    input_angle = input_pose["angle_vec"]
    input_angle_scores = input_pose["angle_score"]
    angles_diff = [np.abs(input_angle[i] - filter_pose_angles[k]) for i,k in enumerate(filter_pose_angles.keys()) if input_angle_scores[i] > 0.1]
    return sum([a > drop_threshold for a in angles_diff]) < 1

def drop_poses(pose_df: pd.DataFrame,
               drop_poses: dict = FILTER_POSES,
               drop_threshold: float = 0.1) -> pd.DataFrame:
     for name,filter_pose in drop_poses.items():
        print(f"Drop {name} poses")
        pose_df = pose_df[pose_df.apply(lambda x: not compare_pose(x, filter_pose_angles=filter_pose, drop_threshold=drop_threshold), axis=1)]
     return pose_df



def add_metadata_to_poses(pose_df: pd.DataFrame) -> pd.DataFrame:
    data = dataframe_from_hdf5(DRIVE_PATH, "metadata")
    data["seq_id"] = data.seq_id.map(lambda x: f"ioc-{x}")
    pose_df = pd.merge(pose_df, data[["seq_id", "sport"]], left_on="media_id", right_on="seq_id")
    return pose_df



def load_local_poses(fp: str) -> pd.DataFrame:
    def parse_list_string(s):
        try:
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            # Handle cases where the string cannot be parsed as a list
            return []

    df = pd.read_csv(fp, converters={"angle_vec": parse_list_string, "angle_score": parse_list_string, "keypoints": parse_list_string, "bbox": parse_list_string})
    return df



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

def process_all_poses(results: list) -> pd.DataFrame:
    """
    Process results from the DB and return a DataFrame with all poses.
    """

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

    return pd.DataFrame(poses_flat) 