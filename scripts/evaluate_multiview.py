"""
Scene Change Detection Evaluation Metrics

This module computes evaluation metrics for scene change detection:
- Dataset-level IoU across all views and frames
- Per-frame Average Precision (AP)
- Per-scene AP without change-type requirement
- Per-scene AP with change-type requirement

The evaluation uses mask IoU-based correspondence with VOC-style AP.
"""

import os
import re
import json
import pickle
import copy
from pathlib import Path
import sys
# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils

from utils import process_video_to_frames


# ============================================================================
# Average Precision Computation
# ============================================================================

def calculate_voc_ap(recalls, precisions):
    """
    Calculate Average Precision using VOC method.
    
    Uses all-point interpolation where precision is set to the maximum
    precision obtained for any recall >= current recall.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        
    Returns:
        Average Precision (float)
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope (maximum precision for recall >= r)
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum (\Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1: First binary mask (torch.Tensor)
        mask2: Second binary mask (torch.Tensor)
        
    Returns:
        IoU value (float)
    """
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()

    mask1 = np.asarray(mask1).astype(bool)
    mask2 = np.asarray(mask2).astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def decode_and_resize_pred_mask(mask_rle, H, W):
    """Decode a prediction mask and resize it to the GT resolution."""
    det_mask = torch.tensor(mask_utils.decode(mask_rle))
    if det_mask.shape[0] != H or det_mask.shape[1] != W:
        det_mask = F.interpolate(
            det_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0) > 0.5
    else:
        det_mask = det_mask > 0.5
    return det_mask.cpu().numpy().astype(bool)


# ============================================================================
# Mask and Bounding Box Utilities
# ============================================================================

def mask_to_bbox(mask):
    """
    Convert a binary mask to a bounding box [x1, y1, x2, y2].
    
    Args:
        mask: Binary mask (numpy array or torch.Tensor)
        
    Returns:
        Bounding box coordinates [x1, y1, x2, y2]
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Convert to uint8
    mask_uint8 = np.uint8(mask) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [0, 0, 0, 0]
    
    # Find bounding box from all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    
    return [x_min, y_min, x_max, y_max]


def center_overlap_detection(pred_center, gt_bbox):
    """
    Check if predicted center point falls within ground truth bounding box.
    
    Args:
        pred_center: Predicted center point (x, y)
        gt_bbox: Ground truth bounding box [x1, y1, x2, y2]
        
    Returns:
        Tuple of (matched, center_distance)
    """
    gt_center = ((gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2)
    
    matched = (gt_bbox[0] < pred_center[0] < gt_bbox[2] and 
               gt_bbox[1] < pred_center[1] < gt_bbox[3])
    
    center_distance = np.linalg.norm(np.array(pred_center) - np.array(gt_center))
    
    return matched, center_distance


def get_mask_medoid_optimized(mask):
    """
    Calculate the medoid point of a binary mask using distance transform.
    
    Finds a point that is central to the mask by computing the point with
    maximum distance from the boundary.
    
    Args:
        mask: Binary mask (torch.Tensor or numpy array)
        
    Returns:
        Tuple (x, y) representing the medoid point, or None if mask is empty
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    if not np.any(mask_np):
        return None
    
    # Calculate distance transform
    dist_transform = ndimage.distance_transform_edt(mask_np)
    
    # Find point with maximum distance from boundary
    y, x = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
    
    return (int(x), int(y))


def get_mask_bbox_center_point(mask):
    """
    Calculate the center point of a mask's bounding box.
    
    Args:
        mask: Binary mask (torch.Tensor or numpy array)
        
    Returns:
        Tuple (x, y) representing the center point
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    bbox = mask_to_bbox(mask_np)
    
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))


def get_target_hw_from_gt(gt, max_length):
    """
    Build a single target (H, W) from GT aspect ratio and max_length.

    Uses the largest available GT frame size from video1/video2 metadata.
    """
    candidate_sizes = []
    for video_key in ['video1_objects', 'video2_objects']:
        video_objects = gt.get(video_key, {})
        for _, frames in video_objects.items():
            for _, frame_meta in frames.items():
                size = frame_meta.get('size', None)
                if size is not None and len(size) >= 2:
                    # GT stores size as (W, H)
                    w, h = int(size[0]), int(size[1])
                    if w > 0 and h > 0:
                        candidate_sizes.append((w, h))
                    break
            if candidate_sizes:
                break

    if not candidate_sizes:
        return None

    src_w, src_h = max(candidate_sizes, key=lambda x: x[0] * x[1])
    scale = float(max_length) / float(max(src_w, src_h))
    target_w = max(1, int(round(src_w * scale)))
    target_h = max(1, int(round(src_h * scale)))
    return target_h, target_w


# ============================================================================
# Visualization
# ============================================================================

def visualize_predictions_and_gt(scene_name, pred_dir, gt_dir, video_dir, 
                                 pred_gt_match_dict, args):
    """
    Create side-by-side visualizations of predictions and ground truth.
    
    Generates comparison images showing predicted and ground truth object masks
    overlaid on the original RGB frames.
    
    Args:
        scene_name: Name of the scene
        pred_dir: Directory containing prediction outputs
        gt_dir: Directory containing ground truth data
        video_dir: Directory containing video files
        pred_gt_match_dict: Dictionary mapping prediction IDs to GT IDs
        args: Arguments containing configurations
    """
    # Define colors for different object types
    three_type_colors = {
        'Removed': (255, 51, 51),    # Red
        'Added': (51, 204, 102),     # Green
        'Moved': (0, 153, 255)       # Blue
    }
    
    # Create visualization directories
    vis_dir_1 = pred_dir / 'video_1_detection'
    vis_dir_2 = pred_dir / 'video_2_detection'
    vis_dir_1.mkdir(parents=True, exist_ok=True)
    vis_dir_2.mkdir(parents=True, exist_ok=True)
    
    # Load prediction and ground truth data
    pred_file = pred_dir / 'object_masks.pkl'
    gt_file = gt_dir / 'segments.pkl'
    
    try:
        pred_data = pickle.load(open(pred_file, 'rb'))
        gt_data = pickle.load(open(gt_file, 'rb'))
    except Exception as e:
        print(f"Error loading data for scene {scene_name}: {e}")
        return
    
    # Get dimensions
    H = pred_data.get('H', 480)
    W = pred_data.get('W', 640)
    
    # Extract prediction and ground truth objects
    pred_objects = {k: v for k, v in pred_data.items() if k not in ['H', 'W']}
    gt_video_1_objects = gt_data['video1_objects']
    gt_video_2_objects = gt_data['video2_objects']
    gt_meta = gt_data.get('objects', {})
    
    # Create color palette
    cmap1 = plt.cm.tab20
    cmap2 = plt.cm.tab20b
    cmap3 = plt.cm.tab20c
    
    colors1 = torch.tensor([cmap1(i)[:3] for i in range(20)]).float()
    colors2 = torch.tensor([cmap2(i)[:3] for i in range(20)]).float()
    colors3 = torch.tensor([cmap3(i)[:3] for i in range(20)]).float()
    colors = torch.cat([colors1, colors2, colors3], dim=0)
    
    # Load RGB frames
    rgb_frames_1, rgb_frames_2 = load_rgb_frames(video_dir, H, W, args)
    
    # Process video 1 frames
    process_video_frames(
        rgb_frames_1, pred_objects, gt_video_1_objects, gt_meta, H, W,
        three_type_colors, vis_dir_1, 'video_1', args, gt_data
    )
    
    # Process video 2 frames
    process_video_frames(
        rgb_frames_2, pred_objects, gt_video_2_objects, gt_meta, H, W,
        three_type_colors, vis_dir_2, 'video_2', args, gt_data
    )
    
    print(f"Visualizations created for scene {scene_name}")


def load_rgb_frames(video_dir, H, W, args):
    """
    Load RGB frames from video directories.
    
    Args:
        video_dir: Directory containing videos
        H, W: Target dimensions
        args: Arguments with resample_rate
        
    Returns:
        Tuple of (rgb_frames_1, rgb_frames_2) dictionaries
    """
    try:
        # Find video files
        video_files = sorted([
            x for x in os.listdir(video_dir) 
            if x.lower().endswith(('.mp4', '.mov', '.avi'))
        ])
        
        if len(video_files) < 2:
            return {}, {}
        
        video_1_path = video_dir / video_files[0]
        video_2_path = video_dir / video_files[1]
        
        video_1_dir = video_dir / 'video1_frames'
        video_2_dir = video_dir / 'video2_frames'
        
        # Extract frames if needed
        if not video_1_dir.exists() or not video_2_dir.exists():
            process_video_to_frames(str(video_1_path), str(video_1_dir))
            process_video_to_frames(str(video_2_path), str(video_2_dir))
        
        # Load frames
        rgb_frames_1 = load_frames_from_dir(video_1_dir, H, W, args.resample_rate)
        rgb_frames_2 = load_frames_from_dir(video_2_dir, H, W, args.resample_rate)
        
        return rgb_frames_1, rgb_frames_2
    
    except Exception as e:
        print(f"Error loading RGB frames: {e}")
        return {}, {}


def load_frames_from_dir(frame_dir, H, W, resample_rate):
    """Load and resize frames from directory."""
    frames = {}
    if frame_dir.exists():
        for img_path in frame_dir.glob('*.jpg'):
            frame_idx = int(img_path.stem.split('_')[-1])
            if frame_idx % resample_rate == 0:
                img = Image.open(img_path)
                img = np.array(img.resize((W, H)))[..., :3] / 255.
                frames[frame_idx // resample_rate] = img
    return frames


def process_video_frames(rgb_frames, pred_objects, gt_objects, gt_meta, H, W,
                        three_type_colors, vis_dir, video_key, args, gt_data=None):
    """
    Process and visualize frames for a single video.
    
    Args:
        rgb_frames: Dictionary of RGB frames
        pred_objects: Predicted objects
        gt_objects: Ground truth objects
        gt_meta: Ground truth metadata
        H, W: Image dimensions
        three_type_colors: Color map for object types
        vis_dir: Visualization output directory
        video_key: 'video_1' or 'video_2'
        args: Configuration arguments
        gt_data: Full ground truth data (optional)
    """
    border = 10
    title_height = 30
    background_color = np.array([1, 1, 1])  # White
    background_weight = 0.4
    
    for frame_idx in sorted(rgb_frames.keys()):
        # Initialize overlays
        overlay_pred = np.zeros((H, W, 3), dtype=np.float32)
        overlay_gt = np.zeros((H, W, 3), dtype=np.float32)
        valid_mask_pred = np.zeros((H, W), dtype=np.uint8)
        valid_mask_gt = np.zeros((H, W), dtype=np.uint8)
        background_mask_pred = np.ones((H, W), dtype=np.uint8)
        background_mask_gt = np.ones((H, W), dtype=np.uint8)
        
        # Base images
        base_img_pred = rgb_frames[frame_idx].copy() if frame_idx in rgb_frames else np.ones((H, W, 3))
        base_img_gt = rgb_frames[frame_idx].copy() if frame_idx in rgb_frames else np.ones((H, W, 3))
        
        # Add predictions
        for pred_id, pred_obj in pred_objects.items():
            if video_key in pred_obj and frame_idx in pred_obj[video_key]:
                add_mask_overlay(
                    pred_obj, frame_idx, video_key, three_type_colors,
                    overlay_pred, valid_mask_pred, background_mask_pred,
                    H, W, is_prediction=True
                )
        
        # Apply prediction overlay
        valid_mask_pred = valid_mask_pred.astype(bool)
        base_img_pred[valid_mask_pred] = (base_img_pred[valid_mask_pred] * 0.5 + 
                                         overlay_pred[valid_mask_pred] * 0.5)
        if args.mask_background:
            bg_mask = background_mask_pred.astype(bool)
            base_img_pred[bg_mask] = (background_color * background_weight + 
                                     base_img_pred[bg_mask] * (1 - background_weight))
        
        # Add ground truth
        if gt_data is not None:
            gt_video_1_objs = gt_data.get('video1_objects', {})
            gt_video_2_objs = gt_data.get('video2_objects', {})
        else:
            gt_video_1_objs = {}
            gt_video_2_objs = {}
        
        for gt_obj_info in gt_meta:
            gt_id = gt_obj_info['original_obj_idx']
            if gt_id in gt_objects:
                orig_frame_idx = str(frame_idx * args.resample_rate)
                if orig_frame_idx in gt_objects[gt_id]:
                    add_gt_mask_overlay(
                        gt_objects[gt_id][orig_frame_idx], gt_id,
                        gt_video_1_objects=gt_video_1_objs,
                        gt_video_2_objects=gt_video_2_objs,
                        three_type_colors=three_type_colors,
                        overlay=overlay_gt, valid_mask=valid_mask_gt,
                        background_mask=background_mask_gt, H=H, W=W
                    )
        
        # Apply GT overlay
        valid_mask_gt = valid_mask_gt.astype(bool)
        base_img_gt[valid_mask_gt] = (base_img_gt[valid_mask_gt] * 0.5 + 
                                     overlay_gt[valid_mask_gt] * 0.5)
        if args.mask_background:
            bg_mask = background_mask_gt.astype(bool)
            base_img_gt[bg_mask] = (background_color * background_weight + 
                                   base_img_gt[bg_mask] * (1 - background_weight))
        
        # Create and save comparison
        canvas = np.ones((H + title_height, 2*W + border, 3))
        canvas[title_height:, 0:W] = base_img_pred
        canvas[title_height:, W+border:] = base_img_gt
        plt.imsave(vis_dir / f"frame_{frame_idx:04d}_comparison.png", canvas)
        
        # Crop if requested
        if args.crop:
            base_img_pred, base_img_gt = crop_images(base_img_pred, base_img_gt, 
                                                     target_size=(500, 600))
        
        # Save individual frames
        plt.imsave(vis_dir / f"pred_frame_{frame_idx:04d}.png", base_img_pred)
        plt.imsave(vis_dir / f"gt_frame_{frame_idx:04d}.png", base_img_gt)


def add_mask_overlay(obj_data, frame_idx, video_key, three_type_colors,
                    overlay, valid_mask, background_mask, H, W, is_prediction=True):
    """Add object mask overlay to visualization."""
    mask_data = obj_data[video_key][frame_idx]
    mask = torch.tensor(mask_utils.decode(mask_data['mask']))
    
    # Resize if needed
    if mask.shape[0] != H or mask.shape[1] != W:
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0) > 0.5
    
    # Determine label and color
    if 'video_1' in obj_data and 'video_2' in obj_data:
        label = 'Moved'
    elif 'video_1' in obj_data:
        label = 'Removed'
    elif 'video_2' in obj_data:
        label = 'Added'
    else:
        label = 'Unknown'
    
    color = np.array(three_type_colors[label]) / 255.
    mask_np = mask.cpu().numpy().astype(bool)
    
    # Apply color overlay
    overlay[mask_np] = color
    valid_mask[mask_np] = True
    
    # Add boundary
    kernel_size = 8 if video_key == 'video_1' else 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask.cpu().numpy().astype(np.uint8), kernel, iterations=1)
    boundary = dilated - mask.cpu().numpy().astype(np.uint8)
    overlay[boundary.astype(bool)] = np.array([0, 0, 0])  # Black boundary
    valid_mask[boundary.astype(bool)] = True
    background_mask[boundary.astype(bool)] = 0
    background_mask[mask_np] = 0


def add_gt_mask_overlay(mask_rle, gt_id, gt_video_1_objects, gt_video_2_objects,
                       three_type_colors, overlay, valid_mask, background_mask, H, W):
    """Add ground truth mask overlay to visualization."""
    gt_mask = torch.tensor(mask_utils.decode(mask_rle))
    
    # Resize if needed
    if gt_mask.shape[0] != H or gt_mask.shape[1] != W:
        gt_mask = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0) > 0.5
    
    # Determine label
    if gt_id in gt_video_1_objects and gt_id in gt_video_2_objects:
        label = 'Moved'
    elif gt_id in gt_video_1_objects:
        label = 'Removed'
    elif gt_id in gt_video_2_objects:
        label = 'Added'
    else:
        label = 'Unknown'
    
    color = np.array(three_type_colors[label]) / 255.
    mask_np = gt_mask.cpu().numpy().astype(bool)
    
    # Apply overlay
    overlay[mask_np] = color
    valid_mask[mask_np] = True
    
    # Add boundary
    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(gt_mask.cpu().numpy().astype(np.uint8), kernel, iterations=1)
    boundary = dilated - gt_mask.cpu().numpy().astype(np.uint8)
    overlay[boundary.astype(bool)] = np.array([0, 0, 0])
    valid_mask[boundary.astype(bool)] = True
    background_mask[boundary.astype(bool)] = 0
    background_mask[mask_np] = 0


def crop_images(img_pred, img_gt, target_size=(500, 600)):
    """
    Crop images to target resolution while preserving aspect ratio.
    
    Args:
        img_pred: Prediction image
        img_gt: Ground truth image
        target_size: Target (width, height)
        
    Returns:
        Tuple of (cropped_pred, cropped_gt)
    """
    current_height, current_width = img_pred.shape[:2]
    current_ratio = current_height / current_width
    target_ratio = target_size[1] / target_size[0]
    
    if current_ratio > target_ratio:
        # Resize width to target, crop height
        new_width = target_size[0]
        new_height = int(current_height * (new_width / current_width))
        
        img_pred_resized = cv2.resize(img_pred, (new_width, new_height))
        img_gt_resized = cv2.resize(img_gt, (new_width, new_height))
        
        center_y = new_height // 2
        start_y = max(0, center_y - target_size[1] // 2)
        end_y = min(new_height, start_y + target_size[1])
        
        img_pred = img_pred_resized[start_y:end_y, 0:new_width]
        img_gt = img_gt_resized[start_y:end_y, 0:new_width]
    else:
        # Resize height to target, crop width
        new_height = target_size[1]
        new_width = int(current_width * (new_height / current_height))
        
        img_pred_resized = cv2.resize(img_pred, (new_width, new_height))
        img_gt_resized = cv2.resize(img_gt, (new_width, new_height))
        
        center_x = new_width // 2
        start_x = max(0, center_x - target_size[0] // 2)
        end_x = min(new_width, start_x + target_size[0])
        
        img_pred = img_pred_resized[0:new_height, start_x:end_x]
        img_gt = img_gt_resized[0:new_height, start_x:end_x]
    
    return img_pred, img_gt


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_all_scenes(valid_scene_names, args, save_result_path, visualize=True):
    """
    Evaluate scene change detection across all scenes.
    
    Computes three metrics:
    1. Global IoU across the whole dataset
    2. Per-frame AP using IoU-based matching (IoU > 0.5)
    3. Per-scene AP with and without class requirement
    
    Args:
        valid_scene_names: List of scene names to evaluate
        args: Configuration arguments
        save_result_path: Path to save evaluation results
        visualize: Whether to generate visualizations
    """
    # Initialize tracking variables
    total_gt_regions_count = 0
    total_gt_objects_count = {}
    all_region_detections_info = []
    all_object_level_detection_info = []
    all_object_level_detection_info_by_label = []
    
    # Global IoU tracking
    global_view_tp = 0.0
    global_view_fp = 0.0
    global_view_fn = 0.0
    
    # Process each scene
    for scene_name in valid_scene_names:
        print(f"\nProcessing scene: {scene_name}")
        
        # Initialize per-scene counters
        total_gt_objects_count[scene_name] = {0: 0, 1: 0}  # non-moved, moved
        
        # Get paths
        pred_dir = get_prediction_dir(args.pred_dir, scene_name)
        gt_dir = Path(args.gt_dir) / scene_name
        video_dir = Path(args.video_dir) / scene_name
        
        # Load data
        try:
            pred_data, gt_data = load_scene_data(pred_dir, gt_dir)
        except Exception as e:
            print(f"Error loading scene {scene_name}: {e}")
            continue
        
        target_hw = get_target_hw_from_gt(gt_data, args.max_length)
        if target_hw is not None:
            H, W = target_hw
        else:
            src_h = pred_data.get('H', 1024)
            src_w = pred_data.get('W', 576)
            scale = float(args.max_length) / float(max(src_h, src_w))
            H = max(1, int(round(src_h * scale)))
            W = max(1, int(round(src_w * scale)))
        
        # Extract ground truth
        scene_gt_objects, scene_gt_labels, scene_gt_objects_by_label = extract_ground_truth(
            gt_data, H, W, args.resample_rate
        )
        total_gt_regions_count += len(scene_gt_objects)
        total_gt_objects_count[scene_name][0] += scene_gt_objects_by_label[0]
        total_gt_objects_count[scene_name][1] += scene_gt_objects_by_label[1]
        
        # Extract detections
        scene_region_detections = extract_detections(pred_data, H, W)
        
        # Compute global IoU
        scene_pred_merged_by_view_frame = {'video_1': {}, 'video_2': {}}
        scene_gt_merged_by_view_frame = {'video_1': {}, 'video_2': {}}
        
        for detection in scene_region_detections:
            video_name = 'video_1' if detection['video'] == 1 else 'video_2'
            frame_idx = detection['frame_idx']
            det_mask_np = detection['mask'].cpu().numpy().astype(bool)
            if frame_idx not in scene_pred_merged_by_view_frame[video_name]:
                scene_pred_merged_by_view_frame[video_name][frame_idx] = np.zeros((H, W), dtype=bool)
            scene_pred_merged_by_view_frame[video_name][frame_idx] = np.logical_or(
                scene_pred_merged_by_view_frame[video_name][frame_idx],
                det_mask_np
            )
        
        for (_, frame_idx, video), gt_mask in scene_gt_objects.items():
            video_name = 'video_1' if video == 1 else 'video_2'
            if frame_idx not in scene_gt_merged_by_view_frame[video_name]:
                scene_gt_merged_by_view_frame[video_name][frame_idx] = np.zeros((H, W), dtype=bool)
            scene_gt_merged_by_view_frame[video_name][frame_idx] = np.logical_or(
                scene_gt_merged_by_view_frame[video_name][frame_idx],
                np.asarray(gt_mask).astype(bool) if isinstance(gt_mask, torch.Tensor) else gt_mask.astype(bool)
            )
        
        for video_name in ['video_1', 'video_2']:
            frame_indices = set(scene_pred_merged_by_view_frame[video_name].keys()) | \
                           set(scene_gt_merged_by_view_frame[video_name].keys())
            for frame_idx in frame_indices:
                pred_mask = scene_pred_merged_by_view_frame[video_name].get(frame_idx, np.zeros((H, W), dtype=bool))
                gt_mask = scene_gt_merged_by_view_frame[video_name].get(frame_idx, np.zeros((H, W), dtype=bool))
                global_view_tp += float(np.logical_and(pred_mask, gt_mask).sum())
                global_view_fp += float(np.logical_and(pred_mask, np.logical_not(gt_mask)).sum())
                global_view_fn += float(np.logical_and(np.logical_not(pred_mask), gt_mask).sum())
        
        pred_data = {k: v for k, v in pred_data.items() if k not in ['H', 'W']}
        # Add labels and confidence to prediction objects
        # Track objects to remove (those without valid video data)
        objects_to_remove = []
        for obj_k, obj_v in pred_data.items():
            # Determine label
            if 'video_1' in obj_v and 'video_2' in obj_v:
                pred_data[obj_k]['label'] = 1  # Moved
            elif 'video_1' in obj_v or 'video_2' in obj_v:
                pred_data[obj_k]['label'] = 0  # Non-moved
            else:
                # Object has no video data - mark for removal
                objects_to_remove.append(obj_k)
                continue
            
            # Compute confidence
            pred_data[obj_k]['confidence'] = compute_object_confidence(obj_v)
        
        # Remove invalid objects
        for obj_k in objects_to_remove:
            del pred_data[obj_k]
        
        # Match detections to ground truth (per-view)
        scene_gt_matched = {k: 0 for k in scene_gt_objects.keys()}
        match_detections_to_gt(
            scene_region_detections, scene_gt_objects, scene_gt_labels,
            scene_gt_matched, all_region_detections_info, scene_name, args
        )
        
        # Object-level evaluation
        evaluate_object_level(
            pred_data, scene_gt_objects, scene_gt_labels,
            all_object_level_detection_info,
            all_object_level_detection_info_by_label,
            scene_name, H, W, args
        )
        
        # Visualize if requested
        if visualize:
            try:
                pred_gt_match_dict = {}  # TODO: Build this from matching
                visualize_predictions_and_gt(
                    scene_name, pred_dir, gt_dir, video_dir,
                    pred_gt_match_dict, args
                )
            except Exception as e:
                print(f"Error visualizing scene {scene_name}: {e}")
    
    # Compute metrics
    metrics = compute_final_metrics(
        all_region_detections_info, total_gt_regions_count,
        all_object_level_detection_info, all_object_level_detection_info_by_label,
        total_gt_objects_count, valid_scene_names,
        global_view_tp, global_view_fp, global_view_fn
    )
    
    # Print and save results
    print_and_save_results(metrics, save_result_path, valid_scene_names, args)


def get_prediction_dir(pred_dir, scene_name):
    """Get prediction directory path."""
    return Path(pred_dir) / scene_name


def load_scene_data(pred_dir, gt_dir):
    """Load prediction and ground truth data for a scene."""
    with open(pred_dir / 'object_masks.pkl', 'rb') as f:
        pred_data = pickle.load(f)
    with open(gt_dir / 'segments.pkl', 'rb') as f:
        gt_data = pickle.load(f)
    return pred_data, gt_data


def extract_ground_truth(gt_data, H, W, resample_rate):
    """
    Extract ground truth objects from dataset as masks.
    
    Returns:
        scene_gt_objects: key: (obj_id, frame_idx, video), value: mask
        scene_gt_labels: key: (obj_id, frame_idx, video), value: label
    """
    gt_video_1_objects = gt_data['video1_objects']
    gt_video_2_objects = gt_data['video2_objects']
    gt_meta = gt_data.get('objects', {})
    
    scene_gt_objects = {}
    scene_gt_labels = {}
    scene_gt_objects_by_label = {0: 0, 1: 0}
    
    for object_info in gt_meta:
        obj_k = object_info['original_obj_idx']
        in_video1 = object_info.get('in_video1', False)
        in_video2 = object_info.get('in_video2', False)
        
        # Determine label: 0=non-moved, 1=moved
        if in_video1 and in_video2:
            label = 1  # Moved
        elif in_video1 or in_video2:
            label = 0  # Non-moved (added or removed)
        else:
            continue
        
        scene_gt_objects_by_label[label] += 1
        
        # Process video 1
        if in_video1 and obj_k in gt_video_1_objects:
            for frame_index, mask in gt_video_1_objects[obj_k].items():
                frame_index = int(frame_index)
                if frame_index % resample_rate == 0:
                    actual_frame_index = int(frame_index // resample_rate)
                    gt_mask = decode_and_resize_mask(mask, H, W)
                    
                    key = (obj_k, actual_frame_index, 1)  # obj_id, frame, video
                    scene_gt_objects[key] = gt_mask
                    scene_gt_labels[key] = label
        
        # Process video 2
        if in_video2 and obj_k in gt_video_2_objects:
            for frame_index, mask in gt_video_2_objects[obj_k].items():
                frame_index = int(frame_index)
                if frame_index % resample_rate == 0:
                    actual_frame_index = int(frame_index // resample_rate)
                    gt_mask = decode_and_resize_mask(mask, H, W)
                    
                    key = (obj_k, actual_frame_index, 2)
                    scene_gt_objects[key] = gt_mask
                    scene_gt_labels[key] = label
    
    return scene_gt_objects, scene_gt_labels, scene_gt_objects_by_label


def decode_and_resize_mask(mask_rle, H, W):
    """Decode RLE mask and resize to target dimensions, return as torch tensor."""
    gt_mask = torch.tensor(mask_utils.decode(mask_rle))
    gt_mask = F.interpolate(
        gt_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W), mode='nearest'
    ).squeeze(0).squeeze(0) > 0.5
    return gt_mask


def extract_detections(pred_data, H, W):
    """Extract detection list from prediction data with resized masks."""
    scene_region_detections = []
    
    for obj_k, obj_v in pred_data.items():
        if obj_k in ['H', 'W']:
            continue
        
        # Process video 1
        if 'video_1' in obj_v:
            for frame_index, mask_dict in obj_v['video_1'].items():
                frame_index = int(frame_index)
                det_mask = torch.tensor(mask_utils.decode(mask_dict['mask']))
                # Resize mask to target dimensions
                if det_mask.shape[0] != H or det_mask.shape[1] != W:
                    det_mask = F.interpolate(
                        det_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W),
                        mode='nearest'
                    ).squeeze(0).squeeze(0) > 0.5
                
                scene_region_detections.append({
                    'obj_id': obj_k,
                    'frame_idx': frame_index,
                    'video': 1,
                    'mask': det_mask,
                    'confidence': mask_dict['cost']
                })
        
        # Process video 2
        if 'video_2' in obj_v:
            for frame_index, mask_dict in obj_v['video_2'].items():
                frame_index = int(frame_index)
                det_mask = torch.tensor(mask_utils.decode(mask_dict['mask']))
                # Resize mask to target dimensions
                if det_mask.shape[0] != H or det_mask.shape[1] != W:
                    det_mask = F.interpolate(
                        det_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(H, W),
                        mode='nearest'
                    ).squeeze(0).squeeze(0) > 0.5
                
                scene_region_detections.append({
                    'obj_id': obj_k,
                    'frame_idx': frame_index,
                    'video': 2,
                    'mask': det_mask,
                    'confidence': mask_dict['cost']
                })
    
    # Sort by confidence
    return sorted(scene_region_detections, key=lambda x: x['confidence'], reverse=True)


def match_detections_to_gt(scene_region_detections, scene_gt_objects, scene_gt_labels,
                           scene_gt_matched, all_region_detections_info, scene_name, args):
    """Match detections to ground truth objects for per-view AP using IoU metric."""
    for detection in scene_region_detections:
        frame_idx = detection['frame_idx']
        video = detection['video']
        det_mask = detection['mask']
        confidence = detection['confidence']
        
        # Find best matching GT using IoU
        max_iou = 0.0
        best_gt_key = None
        
        for gt_key, gt_mask in scene_gt_objects.items():
            if gt_key[1] == frame_idx and gt_key[2] == video:
                iou = compute_iou(det_mask, gt_mask)
                if iou > args.iou_threshold and iou > max_iou:
                    max_iou = iou
                    best_gt_key = gt_key
        
        # Create detection info
        detection_info = {
            'scene_name': scene_name,
            'confidence': confidence,
            'max_iou': max_iou,
            'iou_match': best_gt_key is not None,
            'gt_matched': False,
            'gt_obj_id': None,
            'obj_id': detection['obj_id'],
            'video': video,
            'frame_idx': frame_idx,
            'gt_label': None
        }
        
        # Check if GT was already matched
        add_to_list = True
        if best_gt_key is not None:
            if scene_gt_matched[best_gt_key] == 0:
                # First match - count as true positive
                detection_info['gt_matched'] = True
                detection_info['gt_label'] = scene_gt_labels[best_gt_key]
                detection_info['gt_obj_id'] = best_gt_key[0]
                scene_gt_matched[best_gt_key] = 1
            elif scene_gt_matched[best_gt_key] < args.per_frame_duplicate_match_threshold:
                # Duplicate within threshold - not count as TP but don't add to avoid double counting
                scene_gt_matched[best_gt_key] += 1
                detection_info['gt_matched'] = True
                detection_info['gt_label'] = scene_gt_labels[best_gt_key]
                detection_info['gt_obj_id'] = best_gt_key[0]
                add_to_list = False
            # else: duplicate over threshold - remains as FP (gt_matched=False)
        
        if add_to_list:
            all_region_detections_info.append(detection_info)


def evaluate_object_level(pred_data, scene_gt_objects, scene_gt_labels,
                          all_object_level_detection_info,
                          all_object_level_detection_info_by_label,
                          scene_name, H, W, args):
    """Evaluate object-level metrics (both Type-agnostic and Type-aware)."""
    # Organize GT objects
    gt_objects = organize_gt_objects(scene_gt_objects, scene_gt_labels)
    
    # Prepare prediction objects
    pred_data_filtered = {k: v for k, v in pred_data.items() 
                         if k not in ['H', 'W']}
    
    # Add labels and confidence to prediction objects
    for obj_k, obj_v in pred_data_filtered.items():
        if 'video_1' in obj_v and 'video_2' in obj_v:
            obj_v['label'] = 1  # Moved
        else:
            obj_v['label'] = 0  # Non-moved
        
        obj_v['confidence'] = compute_object_confidence(obj_v)
    
    # Sort by confidence
    pred_objects_sorted = sorted(pred_data_filtered.items(), 
                                key=lambda x: x[1]['confidence'], reverse=True)
    
    # Type-agnostic evaluation (without class requirement)
    evaluate_class_agnostic(
        pred_objects_sorted, gt_objects,
        all_object_level_detection_info, scene_name, args, H, W
    )
    
    # Type-aware evaluation (with class requirement)
    evaluate_class_aware(
        pred_data_filtered, gt_objects,
        all_object_level_detection_info_by_label, scene_name, args, H, W
    )


def organize_gt_objects(scene_gt_objects, scene_gt_labels):
    """Organize ground truth objects by object ID."""
    gt_objects = {}
    
    for (obj_k, frame_idx, video), gt_mask in scene_gt_objects.items():
        video_name = 'video_1' if video == 1 else 'video_2'
        
        if obj_k not in gt_objects:
            gt_objects[obj_k] = {
                'frames': {},
                'label': scene_gt_labels[(obj_k, frame_idx, video)]
            }
        
        if video_name not in gt_objects[obj_k]['frames']:
            gt_objects[obj_k]['frames'][video_name] = {}
        
        gt_objects[obj_k]['frames'][video_name][frame_idx] = gt_mask
    
    return gt_objects


def compute_object_confidence(obj_v):
    """Compute average confidence for an object across all frames."""
    confidences = []
    for video_key in ['video_1', 'video_2']:
        if video_key in obj_v:
            for frame_data in obj_v[video_key].values():
                confidences.append(frame_data['cost'])
    return np.mean(confidences) if confidences else 0


def evaluate_class_agnostic(pred_objects, gt_objects, all_detection_info, scene_name, args, H, W):
    """Evaluate Type-agnostic object-level AP by stacking both videos' frames."""
    scene_gt_matched = {}
    
    for obj_k, obj_v in pred_objects:
        matched_gt_id, max_global_iou = match_object_to_gt_overall_iou(
            obj_v, gt_objects, args, H, W
        )

        detection_info = {
            'scene_name': scene_name,
            'obj_id': obj_k,
            'confidence': obj_v['confidence'],
            'matched': False,
            'global_iou_over_all_frames': max_global_iou
        }
        
        add_to_list = True
        if matched_gt_id is not None:
            if matched_gt_id not in scene_gt_matched:
                scene_gt_matched[matched_gt_id] = 1
                detection_info['matched'] = True
            elif scene_gt_matched[matched_gt_id] < args.duplicate_match_threshold:
                scene_gt_matched[matched_gt_id] += 1
                detection_info['matched'] = True
                add_to_list = False
        
        if add_to_list:
            all_detection_info.append(detection_info)


def evaluate_class_aware(pred_objects, gt_objects, all_detection_info, scene_name, args, H, W):
    """Evaluate Type-aware object-level AP using class matching."""
    scene_gt_matched = {}

    pred_sorted = sorted(pred_objects.items(), key=lambda x: x[1]['confidence'], reverse=True)
    
    for obj_k, obj_v in pred_sorted:
        matched_gt_id, max_global_iou = match_object_to_gt_overall_iou(
            obj_v, gt_objects, args, H, W, required_label=obj_v.get('label', 0)
        )

        detection_info = {
            'scene_name': scene_name,
            'obj_id': obj_k,
            'confidence': obj_v['confidence'],
            'matched': False,
            'global_iou_over_all_frames': max_global_iou
        }
        
        add_to_list = True
        if matched_gt_id is not None:
            if matched_gt_id not in scene_gt_matched:
                scene_gt_matched[matched_gt_id] = 1
                detection_info['matched'] = True
            elif scene_gt_matched[matched_gt_id] < args.duplicate_match_threshold:
                scene_gt_matched[matched_gt_id] += 1
                detection_info['matched'] = True
                add_to_list = False
        
        if add_to_list:
            all_detection_info.append(detection_info)


def build_pred_masks_by_video(obj_v, H, W):
    """Convert one predicted object into per-video frame masks."""
    pred_masks_by_video = {}

    for video_name in ['video_1', 'video_2']:
        if video_name not in obj_v:
            continue
        pred_masks_by_video[video_name] = {}
        for frame_idx, mask_dict in obj_v[video_name].items():
            pred_masks_by_video[video_name][int(frame_idx)] = decode_and_resize_pred_mask(
                mask_dict['mask'], H, W
            )

    return pred_masks_by_video


def compute_stacked_mask_iou(pred_masks_by_video, gt_obj_v, H, W):
    """Stack both sequences' frames and compute one IoU value."""
    total_intersection = 0.0
    total_union = 0.0

    for video_name in ['video_1', 'video_2']:
        pred_frames = pred_masks_by_video.get(video_name, {})
        gt_frames = gt_obj_v['frames'].get(video_name, {})
        frame_indices = set(pred_frames.keys()) | set(gt_frames.keys())

        for frame_idx in frame_indices:
            pred_mask = pred_frames.get(frame_idx, np.zeros((H, W), dtype=bool))
            gt_mask = gt_frames.get(frame_idx, np.zeros((H, W), dtype=bool))
            total_intersection += float(np.logical_and(pred_mask, gt_mask).sum())
            total_union += float(np.logical_or(pred_mask, gt_mask).sum())

    return float(total_intersection / total_union) if total_union > 0 else 0.0


def match_object_to_gt_overall_iou(obj_v, gt_objects, args, H, W, required_label=None):
    """Match one predicted object to one GT object using stacked IoU."""
    pred_masks_by_video = build_pred_masks_by_video(obj_v, H, W)
    if not pred_masks_by_video:
        return None, 0.0

    matched_gt_id = None
    max_global_iou = 0.0

    for gt_obj_id, gt_obj_v in gt_objects.items():
        if required_label is not None and gt_obj_v['label'] != required_label:
            continue

        global_iou = compute_stacked_mask_iou(pred_masks_by_video, gt_obj_v, H, W)
        if global_iou > max_global_iou and global_iou > args.iou_threshold:
            max_global_iou = global_iou
            matched_gt_id = gt_obj_id

    return matched_gt_id, max_global_iou


def compute_final_metrics(all_region_detections_info, total_gt_regions_count,
                         all_object_level_detection_info,
                         all_object_level_detection_info_by_label,
                         total_gt_objects_count, valid_scene_names,
                         global_view_tp, global_view_fp, global_view_fn):
    """Compute final metrics: global IoU, per-frame AP, and per-scene AP."""
    
    # Metric 1: Global IoU across all frames and views
    global_view_union = global_view_tp + global_view_fp + global_view_fn
    global_iou = float(global_view_tp / global_view_union) if global_view_union > 0 else 0.0
    
    # Metric 2: Per-frame AP using IoU-based matching
    all_region_detections_info = sorted(all_region_detections_info, key=lambda x: x['confidence'], reverse=True)
    tp = np.array([1 if d['gt_matched'] else 0 for d in all_region_detections_info])
    fp = np.array([0 if d['gt_matched'] else 1 for d in all_region_detections_info])
    
    tp_cumsum = np.cumsum(tp) if len(tp) > 0 else np.array([])
    fp_cumsum = np.cumsum(fp) if len(fp) > 0 else np.array([])
    recalls = tp_cumsum / total_gt_regions_count if total_gt_regions_count > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum) if len(tp_cumsum) > 0 else np.zeros_like(tp_cumsum)
    per_frame_ap = calculate_voc_ap(recalls, precisions)
    
    # Metric 3a: Per-scene AP without class requirement (type-agnostic)
    all_object_level_detection_info = sorted(all_object_level_detection_info, 
                                             key=lambda x: x['confidence'], reverse=True)
    tp_obj = np.array([1 if d['matched'] else 0 for d in all_object_level_detection_info])
    fp_obj = np.array([0 if d['matched'] else 1 for d in all_object_level_detection_info])
    
    total_gt_objects_all = sum(sum(total_gt_objects_count[s].values()) for s in valid_scene_names)
    
    tp_obj_cumsum = np.cumsum(tp_obj) if len(tp_obj) > 0 else np.array([])
    fp_obj_cumsum = np.cumsum(fp_obj) if len(fp_obj) > 0 else np.array([])
    recalls_obj = tp_obj_cumsum / total_gt_objects_all if total_gt_objects_all > 0 else np.zeros_like(tp_obj_cumsum)
    precisions_obj = tp_obj_cumsum / (tp_obj_cumsum + fp_obj_cumsum) if len(tp_obj_cumsum) > 0 else np.zeros_like(tp_obj_cumsum)
    per_scene_ap_without_class = calculate_voc_ap(recalls_obj, precisions_obj)
    
    # Metric 3b: Per-scene AP with class requirement (type-aware)
    all_object_level_detection_info_by_label = sorted(all_object_level_detection_info_by_label,
                                                      key=lambda x: x['confidence'], reverse=True)
    tp_label = np.array([1 if d['matched'] else 0 for d in all_object_level_detection_info_by_label])
    fp_label = np.array([0 if d['matched'] else 1 for d in all_object_level_detection_info_by_label])
    
    total_gt_objects_by_label = sum(sum(total_gt_objects_count[s].values()) for s in valid_scene_names)
    
    tp_label_cumsum = np.cumsum(tp_label) if len(tp_label) > 0 else np.array([])
    fp_label_cumsum = np.cumsum(fp_label) if len(fp_label) > 0 else np.array([])
    recalls_label = tp_label_cumsum / total_gt_objects_by_label if total_gt_objects_by_label > 0 else np.zeros_like(tp_label_cumsum)
    precisions_label = tp_label_cumsum / (tp_label_cumsum + fp_label_cumsum) if len(tp_label_cumsum) > 0 else np.zeros_like(tp_label_cumsum)
    per_scene_ap_with_class = calculate_voc_ap(recalls_label, precisions_label)
    
    return {
        'global_iou': global_iou,
        'per_frame_ap': per_frame_ap,
        'per_scene_ap_without_class': per_scene_ap_without_class,
        'per_scene_ap_with_class': per_scene_ap_with_class,
        'tp': tp_obj_cumsum[-1] if len(tp_obj_cumsum) > 0 else 0,
        'fp': fp_obj_cumsum[-1] if len(fp_obj_cumsum) > 0 else 0,
        'fn': total_gt_objects_all - (tp_obj_cumsum[-1] if len(tp_obj_cumsum) > 0 else 0),
        'global_view_tp': global_view_tp,
        'global_view_fp': global_view_fp,
        'global_view_fn': global_view_fn
    }


def print_and_save_results(metrics, save_path, valid_scene_names, args):
    """Print and save evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Number of scenes: {len(valid_scene_names)}")
    print(f"IoU threshold: {args.iou_threshold}\n")
    
    # Metric 1: Global IoU
    print(f"Metric 1: px/im IoU (merged masks, all views): {metrics['global_iou']:.4f}")
    print(f"  TP: {metrics['global_view_tp']:.0f}, FP: {metrics['global_view_fp']:.0f}, FN: {metrics['global_view_fn']:.0f}\n")
    
    # Metric 2: Per-frame AP
    print(f"Metric 2: obj/im AP (IoU-based matching): {metrics['per_frame_ap']:.4f}\n")
    
    # Metric 3: Per-scene AP
    print(f"Metric 3a: obj/sc AP (without class requirement): {metrics['per_scene_ap_without_class']:.4f}")
    print(f"Metric 3b: obj/sc AP (with class requirement): {metrics['per_scene_ap_with_class']:.4f}")
    
    print(f"\nDetection Statistics:")
    print(f"  True Positives: {metrics['tp']:.0f}")
    print(f"  False Positives: {metrics['fp']:.0f}")
    print(f"  False Negatives: {metrics['fn']:.0f}")
    print("="*80 + "\n")
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(f"Scenes: {len(valid_scene_names)}\n")
        f.write(f"Duplicate match threshold: {args.duplicate_match_threshold}\n")
        f.write(f"Per-frame duplicate threshold: {args.per_frame_duplicate_match_threshold}\n")
        f.write(f"IoU threshold: {args.iou_threshold}\n\n")
        
        f.write(f"Metric 1: px/im IoU (merged masks, all views): {metrics['global_iou']:.4f}\n")
        f.write(f"  TP: {metrics['global_view_tp']:.0f}, FP: {metrics['global_view_fp']:.0f}, FN: {metrics['global_view_fn']:.0f}\n\n")
        
        f.write(f"Metric 2: obj/im AP (IoU-based matching): {metrics['per_frame_ap']:.4f}\n\n")
        
        f.write(f"Metric 3a: obj/sc AP (without class requirement): {metrics['per_scene_ap_without_class']:.4f}\n")
        f.write(f"Metric 3b: obj/sc AP (with class requirement): {metrics['per_scene_ap_with_class']:.4f}\n\n")
        
        f.write(f"Detection Statistics:\n")
        f.write(f"  True Positives: {metrics['tp']:.0f}\n")
        f.write(f"  False Positives: {metrics['fp']:.0f}\n")
        f.write(f"  False Negatives: {metrics['fn']:.0f}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate scene change detection predictions'
    )
    
    # I/O arguments
    parser.add_argument('--pred_dir', type=str, default='output/scenediff_benchmark/',
                       help='Directory containing prediction outputs')
    parser.add_argument('--gt_dir', type=str, 
                       default='data/scenediff_benchmark/data',
                       help='Directory containing ground truth data')
    parser.add_argument('--video_dir', type=str,
                       default='data/scenediff_benchmark/data',
                       help='Directory containing video files')
    parser.add_argument('--output_path', type=str, default='output/result.txt',
                       help='Output filename')
    
    # Evaluation parameters
    parser.add_argument('--resample_rate', type=int, default=30,
                       help='Frame sampling rate')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Longest image side used when resizing GT-derived target resolution')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for IoU-based matching')
    parser.add_argument('--duplicate_match_threshold', type=int, default=1,
                       help='Max matches per GT object (object-level)')
    parser.add_argument('--per_frame_duplicate_match_threshold', type=int, default=1,
                       help='Max matches per GT detection (per-view)')
    
    # Dataset selection
    parser.add_argument('--splits', type=str, default='all', 
                       choices=['val', 'test', 'all'],
                       help='Dataset split to evaluate')
    parser.add_argument('--sets', type=str, default='all',
                       choices=['varied', 'kitchen', 'all'],
                       help='Dataset subset to evaluate')
    
    # Visualization options
    parser.add_argument('--visualize', type=eval, default=False,
                       help='Generate visualizations')
    parser.add_argument('--mask_background', type=eval, default=False,
                       help='Mask background in visualizations')
    parser.add_argument('--crop', type=eval, default=False,
                       help='Crop visualizations for paper figures')
    
    args = parser.parse_args()
    
    # Load dataset splits
    val_split = json.load(open('data/scenediff_benchmark/splits/val_split.json', 'r'))
    val_split['all'] = val_split['varied'] + val_split['kitchen']
    
    test_split = json.load(open('data/scenediff_benchmark/splits/test_split.json', 'r'))
    test_split['all'] = test_split['varied'] + test_split['kitchen']
    
    # Determine scenes to evaluate
    if args.splits == 'all':
        valid_scene_names = val_split[args.sets] + test_split[args.sets]
    elif args.splits == 'val':
        valid_scene_names = val_split[args.sets]
    elif args.splits == 'test':
        valid_scene_names = test_split[args.sets]
    else:
        raise ValueError(f"Invalid split: {args.splits}")
    
    # Filter to only predicted scenes
    not_predicted = []
    
    for scene_name in valid_scene_names:
        scene_dir = Path(args.pred_dir) / scene_name
        if not (scene_dir / 'object_masks.pkl').exists():
            not_predicted.append(scene_name)
    
    for scene_name in not_predicted:
        valid_scene_names.remove(scene_name)
        
    print(f"\nEvaluating {len(valid_scene_names)} scenes")
    print(f"Not predicted: {len(not_predicted)} scenes")
    
    # Evaluate all scenes
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_result_path = output_path.parent / (
        output_path.stem + 
        f'_frame_dup_{args.per_frame_duplicate_match_threshold}_obj_dup_{args.duplicate_match_threshold}' +
        output_path.suffix
    )
    
    evaluate_all_scenes(valid_scene_names, args, save_result_path, visualize=args.visualize)
    
    # Evaluate per scene
    for scene_name in valid_scene_names:
        scene_result_path = os.path.join(
            args.pred_dir, scene_name,
            f'eval_result.txt'
        )
        
    #     evaluate_all_scenes([scene_name], args, scene_result_path, visualize=False)


if __name__ == "__main__":
    main()
