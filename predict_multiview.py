"""
SceneDiff: Multi-view Scene Change Detection

This module implements a multi-view scene change detection pipeline that:
1. Processes pairs of video sequences
2. Estimates depth and camera poses using Pi3
3. Detects changed objects between scenes
4. Associates objects across views using geometric and appearance features

Pipeline Overview:
-----------------
Stage 1:  Load and preprocess images
Stage 2:  Estimate depth maps and camera poses (Pi3)
Stage 3:  Compute inter-view similarity matrix
Stage 4:  Compute per-image confidence maps
Stage 5:  Generate/load SAM segmentation masks
Stage 6:  Load DINOv3 feature extractor
Stage 7:  Compute multi-view cost maps (geometric + appearance)
Stage 8:  Voxelize point clouds and aggregate costs
Stage 9:  Extract DINOv3 features per voxel
Stage 10: Visualize cost maps as point clouds (optional)
Stage 11: Detect and merge objects across frames
Stage 12: Associate objects between videos and save masks
Stage 13: Generate ground truth visualizations (if available)

Key Components:
--------------
- Geometric costs: Foreground distance based on depth
- Appearance costs: DINOv3 feature matching and rendering consistency
- Object detection: Automatic thresholding (Otsu or max entropy)
- Object merging: Geometric and visual similarity across frames
- Object association: Cross-video matching for change detection
"""

import sys
import os
import json
import pickle
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils
from skimage.filters import threshold_otsu
from torch_scatter import scatter_mean
import open3d as o3d

# Add submodules to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PI3_PATH = os.path.join(PROJECT_ROOT, 'submodules/Pi3')
SEGMENT_LS_PATH = os.path.join(PROJECT_ROOT, 'submodules/segment_ls')
DINOV3_PATH = os.path.join(PROJECT_ROOT, 'submodules/dinov3')
sys.path.append(PI3_PATH)
sys.path.append(SEGMENT_LS_PATH)

from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from segment_anything_ls import build_sam, SamAutomaticMaskGenerator

# Import all utilities from utils_two_view
from utils import (
    load_images_as_tensor_from_list,
    process_video_to_frames,
    recover_focal_shift,
    voxelization,
    get_robust_voxel_size,
    get_img_coor,
    valid_mask_after_proj,
    foreground_distance,
    calculate_mask_percentage,
    reprojected_feature,
    predict_dinov3_feat,
    predict_mask,
    apply_colormap,
    compute_conf_norm_distance,
    threshold_maximum_entropy,
    reorder_mask,
    mask_to_rle_pytorch,
    coco_encode_rle,
    merge_objects,
    get_dino_matched_region_cost,
    remove_small_masks
)


# ============================================================================
# Configuration and Constants
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
APPEARANCE_FEATURE_DIM = 1280

# Make paths configurable via environment variables
SAM_CHECKPOINT = os.getenv('SAM_CHECKPOINT', "/u/ywu20/sam_vit_h_4b8939.pth")


# ============================================================================
# Utility Functions
# ============================================================================

def per_single_img_confidence(single_image, model, threshold=0.5, dtype=torch.bfloat16, return_depth=False):
    """
    Compute confidence map for a single image using the depth estimation model.
    
    Args:
        single_image: Input image tensor (C, H, W)
        model: Depth estimation model
        threshold: Confidence threshold percentile
        dtype: Data type for inference
        return_depth: Whether to return depth map along with confidence
        
    Returns:
        Confidence map (H, W) and optionally depth map
    """
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(single_image[None])
    
    depth_conf = torch.sigmoid(res['conf'][..., 0])
    depth_map = res['local_points'][..., 2]
    
    conf_max = torch.kthvalue(
        depth_conf.reshape(-1), 
        int(threshold * depth_conf.reshape(-1).numel())
    ).values.item()
    
    H, W = depth_conf.shape[-2:]
    conf = (depth_conf / conf_max).clamp(0, 1).view(H, W)
    
    if return_depth:
        return conf, depth_map
    return conf


# ============================================================================
# Image and Depth Processing
# ============================================================================

def load_and_preprocess_images(file_list, device):
    """Load images from file list and preprocess them."""
    images = load_images_as_tensor_from_list(file_list)[None].to(device)
    return images


def estimate_depth_and_poses(images, model):
    """
    Estimate depth maps and camera poses for input images.
    
    Args:
        images: Input images (B, N, C, H, W)
        model: Pi3 depth estimation model
        
    Returns:
        Dictionary containing depth, poses, intrinsics, and point maps
    """
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(images)
    
    depth_map = res['local_points'][..., 2][..., None]
    depth_conf = torch.sigmoid(res['conf'][..., 0])
    point_map = res['points']
    poses = res['camera_poses']
    
    return {
        'depth_map': depth_map,
        'depth_conf': depth_conf,
        'point_map': point_map,
        'poses': poses,
        'local_points': res['local_points'],
        'conf': res['conf']
    }


def compute_intrinsics(res, file_list, H, W):
    """
    Compute camera intrinsics for each view.
    
    Args:
        res: Model output dictionary
        file_list: List of image files
        H, W: Image dimensions
        
    Returns:
        Intrinsic matrices (1, N, 3, 3)
    """
    intrinsics = []
    for i in range(len(file_list)):
        local_point_map = res['local_points'][0, i]
        masks = torch.sigmoid(res['conf'][0, i]) > 0.1
        
        original_height, original_width = local_point_map.shape[-3:-1]
        aspect_ratio = original_width / original_height
        
        focal, shift = recover_focal_shift(local_point_map, masks)
        
        fx = focal * W / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal * H / 2 * (1 + aspect_ratio ** 2) ** 0.5
        
        K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]]).cuda()
        intrinsics.append(K)
    
    return torch.stack(intrinsics, dim=0)[None]


def normalize_scene_scale(point_map, depth_map, poses, subsample_size=1000000, total_voxel_number=200):
    """
    Normalize scene scale using voxel size estimation.
    
    Args:
        point_map: Point cloud (N, H, W, 3)
        depth_map: Depth maps (B, N, H, W, 1)
        poses: Camera poses (B, N, 4, 4)
        
    Returns:
        Tuple of (scaled_depth, scaled_points, scaled_poses, voxel_size)
    """
    voxel_size = get_robust_voxel_size(
        point_map.reshape(-1, 3), 
        subsample_size=subsample_size, 
        scale_factor=total_voxel_number
    )
    scale = 1 / (voxel_size * total_voxel_number)
    
    depth_scaled = depth_map * scale
    points_scaled = point_map * scale
    poses_scaled = poses.clone()
    poses_scaled[:, :, :3, 3] = poses_scaled[:, :, :3, 3] * scale
    
    return depth_scaled, points_scaled, poses_scaled, voxel_size


# ============================================================================
# Similarity and Cost Computation
# ============================================================================

def compute_similarity_matrix(poses, depth_map, intrinsic, grid_intrinsic, img_coors, 
                              H, W, img_1_length, img_2_length):
    """
    Compute pairwise similarity between images from two sets.
    
    Args:
        poses: Camera poses (B, N, 4, 4)
        depth_map: Depth maps (B, N, H, W, 1)
        intrinsic: Intrinsic matrices (B, N, 3, 3)
        grid_intrinsic: Grid intrinsic for projection
        img_coors: Image coordinates
        H, W: Image dimensions
        img_1_length: Number of images in first set
        img_2_length: Number of images in second set
        
    Returns:
        Similarity matrix (N, N)
    """
    n_total = img_1_length + img_2_length
    similarity_matrix = -1 * torch.ones(n_total, n_total).cuda()
    
    for i in range(img_1_length):
        for j in range(img_2_length):
            set_2_idx = img_1_length + j
            
            sim_1 = calculate_mask_percentage(
                poses[0][i], poses[0][set_2_idx],
                depth_map[0][i].permute(2,0,1), depth_map[0][set_2_idx].permute(2,0,1),
                intrinsic[0][i], intrinsic[0][set_2_idx],
                grid_intrinsic, 0.3, img_coors, H, W
            )
            
            sim_2 = calculate_mask_percentage(
                poses[0][set_2_idx], poses[0][i],
                depth_map[0][set_2_idx].permute(2,0,1), depth_map[0][i].permute(2,0,1),
                intrinsic[0][set_2_idx], intrinsic[0][i],
                grid_intrinsic, 0.3, img_coors, H, W
            )
            
            similarity_matrix[i, set_2_idx] = (sim_1 + sim_2) / 2
            similarity_matrix[set_2_idx, i] = similarity_matrix[i, set_2_idx]
    
    return similarity_matrix


def compute_similarity_weights(similarity_matrix, percentage_threshold=0.5):
    """
    Convert similarity matrix to weights for aggregation.
    
    Args:
        similarity_matrix: Pairwise similarities (N, N)
        percentage_threshold: Minimum similarity threshold
        
    Returns:
        Normalized weight matrix (N, N)
    """
    n = similarity_matrix.shape[0]
    weight_matrix = torch.zeros_like(similarity_matrix)
    
    for i in range(n):
        for j in range(n):
            if similarity_matrix[i, j] > percentage_threshold:
                weight_matrix[i, j] = similarity_matrix[i, j]
        
        if weight_matrix[i].sum() == 0:
            j = torch.argmax(similarity_matrix[i])
            weight_matrix[i, j] = 1
    
    weight_matrix = weight_matrix / weight_matrix.sum(dim=1, keepdim=True)
    return weight_matrix


def compute_confidence_maps(images, model, file_list):
    """Compute confidence maps for all images."""
    conf_list = []
    for i in range(len(file_list)):
        conf = per_single_img_confidence(images[0][i:i+1].clone(), model,)
        conf_list.append([conf])
    return conf_list


# ============================================================================
# SAM Mask Generation
# ============================================================================

def load_or_generate_sam_masks(file_list, H, W, mask_file_path):
    """
    Load cached SAM masks or generate new ones.
    
    Args:
        file_list: List of image paths
        H, W: Target dimensions
        mask_file_path: Path to cached masks
        
    Returns:
        List of non-overlapping masks with each value being the index of the mask and -1 for no mask
    """
    if os.path.exists(mask_file_path):
        with open(mask_file_path, 'rb') as f:
            masks_list = pickle.load(f)
        
        # Verify and resize if needed
        for i, mask in enumerate(masks_list):
            if mask.shape != (H, W):
                mask = cv2.resize(
                    mask.astype(np.float32), 
                    (W, H), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int64)
                masks_list[i] = reorder_mask(mask)
        
        return masks_list
    
    # Generate new masks,
    mask_generator = SamAutomaticMaskGenerator(
        model=build_sam(checkpoint=SAM_CHECKPOINT).to(device=DEVICE),
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    
    masks_list = []
    for filepath in file_list:
        image = np.array(Image.open(filepath)).astype(np.uint8)
        mask = predict_mask(image, mask_generator)
        mask = cv2.resize(mask.astype(np.float32), (W, H), 
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
        # Still loop to remove the mask that is too small
        mask = remove_small_masks(mask, )
        masks_list.append(reorder_mask(mask))
    
    # Cache the masks
    with open(mask_file_path, 'wb') as f:
        pickle.dump(masks_list, f)
    
    del mask_generator
    torch.cuda.empty_cache()
    
    return masks_list


# ============================================================================
# Cost Map Computation
# ============================================================================

def compute_cost_maps_for_image(index, images, masks_list, poses, depth_map, intrinsic,
                                conf_list, similarity_weights, pi3_masks, img_coors,
                                grid_intrinsic, dinov3_model, H, W, args):
    """
    Compute cost maps for a single image by aggregating costs from similar views.
    
    Args:
        index: Current image index
        images: All images
        masks_list: SAM masks for all images
        poses: Camera poses
        depth_map: Depth maps
        intrinsic: Camera intrinsics
        conf_list: Confidence maps
        similarity_weights: Weight matrix for view aggregation
        pi3_masks: Valid depth masks
        img_coors: Image coordinates
        grid_intrinsic: Grid intrinsic matrix
        dinov3_model: DINOv3 feature extractor
        H, W: Image dimensions
        args: Configuration arguments
        
    Returns:
        Tuple of (weighted_cost_map, visible_map)
    """
    cost_map_list = []
    visible_map_list = []
    non_occluded_map_list = []
    view_coverage_map_list = []
    
    img1_np = images[0][index].permute(1,2,0).cpu().numpy()
    sam_mask_1 = torch.tensor(masks_list[index]).cuda()
    pose_1 = poses[0][index].clone()
    depth_1 = depth_map[0][index].view(1, H, W).clone()
    intrinsic_1 = intrinsic[0][index].clone()
    conf_1 = conf_list[index][0]
    
    for closest_index in similarity_weights[index].argsort(dim=0, descending=True):
        if similarity_weights[index, closest_index] == 0:
            break
        
        img2_np = images[0][closest_index].permute(1,2,0).cpu().numpy()
        sam_mask_2 = torch.tensor(masks_list[closest_index]).cuda()
        pose_2 = poses[0][closest_index].clone()
        depth_2 = depth_map[0][closest_index].view(1, H, W).clone()
        intrinsic_2 = intrinsic[0][closest_index].clone()
        conf_2 = conf_list[closest_index][0]
        
        # Compute visibility and foreground distances
        vis_1 = valid_mask_after_proj(intrinsic_1.cuda(), pose_1.cuda(), depth_1,
                                      intrinsic_2.cuda(), pose_2.cuda(), depth_2,
                                      img_coors, H, W)
        vis_2 = valid_mask_after_proj(intrinsic_2.cuda(), pose_2.cuda(), depth_2,
                                      intrinsic_1.cuda(), pose_1.cuda(), depth_1,
                                      img_coors, H, W)
        
        foreground_dis_1 = foreground_distance(intrinsic_1.cuda(), pose_1.cuda(), depth_1,
                                               intrinsic_2.cuda(), pose_2.cuda(), depth_2,
                                               img_coors, grid_intrinsic, H, W)
        foreground_dis_2 = foreground_distance(intrinsic_2.cuda(), pose_2.cuda(), depth_2,
                                               intrinsic_1.cuda(), pose_1.cuda(), depth_1,
                                               img_coors, grid_intrinsic, H, W)
        
        # Compute geometric costs
        fg_pixel_geo_conf_1 = compute_conf_norm_distance(foreground_dis_1, conf_1)
        fg_pixel_geo_conf_2 = compute_conf_norm_distance(foreground_dis_2, conf_2)
        
        # Aggregate to regions
        fg_region_geo_1 = aggregate_to_regions(fg_pixel_geo_conf_1 * vis_1, sam_mask_1)
        fg_region_geo_2 = aggregate_to_regions(fg_pixel_geo_conf_2 * vis_2, sam_mask_2)
        
        # Compute DINOv3 features and costs
        dinov3_feat_1 = predict_dinov3_feat(img1_np, dinov3_model)
        dinov3_feat_2 = predict_dinov3_feat(img2_np, dinov3_model)
        
        occlusion_mask_1 = (foreground_dis_1 < args.occlusion_threshold) | \
                           ~pi3_masks[index] | ~vis_1
        occlusion_mask_2 = (foreground_dis_2 < args.occlusion_threshold) | \
                           ~pi3_masks[closest_index] | ~vis_2
        
        # DINOv3 region matching cost
        dino_region_match_1, dino_region_match_2, _, _, _, _ = get_dino_matched_region_cost(
            dinov3_feat_1, dinov3_feat_2, sam_mask_1, sam_mask_2,
            ~occlusion_mask_1, ~occlusion_mask_2, foreground_dis_1, foreground_dis_2
        )
        
        # DINOv3 rendering cost
        dino_render_cost_1 = compute_rendering_cost(
            dinov3_feat_1, dinov3_feat_2, intrinsic_1, pose_1, depth_1,
            intrinsic_2, pose_2, img_coors, grid_intrinsic, H, W,
            occlusion_mask_1, sam_mask_1, vis_1, args
        )
        dino_render_cost_2 = compute_rendering_cost(
            dinov3_feat_2, dinov3_feat_1, intrinsic_2, pose_2, depth_2,
            intrinsic_1, pose_1, img_coors, grid_intrinsic, H, W,
            occlusion_mask_2, sam_mask_2, vis_2, args
        )
        
        # Stack costs and visibility masks
        costs = torch.stack([
            (fg_region_geo_1 * vis_1).cpu(),
            (dino_region_match_1 * vis_1).cpu(),
            (dino_render_cost_1 * vis_1).cpu()
        ], dim=0)
        
        cost_map_list.append(costs)
        visible_map_list.append((~occlusion_mask_1 & vis_1).cpu())
        non_occluded_map_list.append(~occlusion_mask_1.cpu())
        view_coverage_map_list.append(vis_1.cpu())
    
    # Aggregate costs across views
    cost_maps = torch.stack(cost_map_list)  # (N_views, 3, H, W)
    visible_maps = torch.stack(visible_map_list)  # (N_views, H, W)
    
    visible_count = visible_maps.sum(dim=0)
    weighted_cost = (cost_maps * visible_maps[:, None, :, :]).sum(dim=0)
    weighted_cost = weighted_cost / (visible_count + 1e-6)[None, :, :]
    weighted_cost[visible_count[None].repeat(3, 1, 1) == 0] = 0
    
    return weighted_cost, visible_count > 0


def aggregate_to_regions(pixel_values, mask):
    """Aggregate pixel-level values to region-level using mask."""
    region_values = torch.zeros_like(mask).float()
    for region_id in mask.unique():
        if region_id == -1:
            continue
        region_mask = mask == region_id
        region_values[region_mask] = pixel_values[region_mask].mean()
    return region_values


def compute_rendering_cost(feat_src, feat_tgt, intrinsic_src, pose_src, depth_src,
                           intrinsic_tgt, pose_tgt, img_coors, grid_intrinsic, H, W,
                           occlusion_mask, sam_mask, vis_mask, args):
    """Compute rendering-based appearance cost."""
    feat_reprojected = reprojected_feature(
        intrinsic_tgt.cuda(), pose_tgt.cuda(), depth_src.view(1, H, W),
        intrinsic_src.cuda(), pose_src.cuda(), feat_tgt,
        img_coors, grid_intrinsic, H, W
    )
    
    diff = 1 - torch.cosine_similarity(feat_reprojected, feat_src, dim=0)
    diff[occlusion_mask] = 0
    
    region_diff = torch.zeros_like(sam_mask).float()
    for region_id in sam_mask.unique():
        if region_id == -1:
            continue
        region_mask = sam_mask == region_id
        
        valid_mask = region_mask & ~occlusion_mask
        
        if valid_mask.sum() > 0:
            region_diff[region_mask] = diff[valid_mask].mean()
    
    return region_diff


# ============================================================================
# Object Detection and Merging
# ============================================================================

def detect_and_threshold_objects(total_cluster_cost_maps, total_visible_maps,
                                 masks_list, pc_voxel_locations, pc_voxel_indices,
                                 dino_features, img_length, H, W, args):
    """
    Detect objects by thresholding cost maps.
    
    Args:
        total_cluster_cost_maps: Cost maps aggregated to regions (N, H, W)
        total_visible_maps: Visibility masks (N, H, W)
        masks_list: SAM segmentation masks
        pc_voxel_locations: Voxel center locations
        pc_voxel_indices: Point-to-voxel mapping
        dino_features: DINOv3 features per voxel
        img_length: Number of images
        H, W: Image dimensions
        args: Configuration arguments
        
    Returns:
        Tuple of (object_list, mask_indices, threshold)
    """
    # Get visible cost values
    visible_costs = total_cluster_cost_maps[total_visible_maps]
    
    # Compute threshold
    if args.change_region_threshold is not None:
        threshold = args.change_region_threshold
    else:
        quantile_val = torch.quantile(visible_costs, args.filter_percentage_before_threshold)
        filtered_costs = visible_costs[visible_costs > quantile_val]
        
        if args.threshold_method == 'otsu':
            threshold = threshold_otsu(filtered_costs.view(-1)[None].cpu().numpy())
        elif args.threshold_method == 'max_entropy':
            threshold = threshold_maximum_entropy(filtered_costs)
        else:
            raise ValueError(f"Invalid threshold method: {args.threshold_method}")
    
    # Threshold to get object masks
    object_masks = total_cluster_cost_maps > threshold
    mask_indices = -1 * torch.ones_like(object_masks).int()
    object_list = []
    mask_counter = 0
    
    for idx in range(img_length):
        image_objects = []
        sam_mask = torch.tensor(masks_list[idx]).cuda()
        sam_mask[~object_masks[idx]] = -1
        
        for region_id in sam_mask.unique():
            if region_id == -1:
                continue
            
            region_mask = sam_mask == region_id
            if region_mask.sum() < args.min_detection_pixel:
                continue
            
            # Extract voxel points and features
            voxel_ids = pc_voxel_indices.view(-1, H, W)[idx][region_mask]
            points = pc_voxel_locations[voxel_ids].unique(dim=0).view(-1, 3)
            feat = dino_features[voxel_ids.unique()].mean(dim=0)
            
            obj_dict = {
                'pc': points,
                'dino_feat': feat
            }
            image_objects.append(obj_dict)
            mask_indices[idx][region_mask] = mask_counter
            mask_counter += 1
        
        object_list.append(image_objects)
    
    return object_list, mask_indices, threshold


def create_unified_object_ids(object_id_list_1, object_id_list_2, object_sim_matrix, args):
    """
    Create unified object IDs across two video sequences.
    
    Objects with high similarity are assigned the same ID.
    
    Args:
        object_id_list_1: Object IDs from video 1
        object_id_list_2: Object IDs from video 2
        object_sim_matrix: Similarity matrix between objects
        args: Configuration with similarity threshold
        
    Returns:
        Dictionary with 'video_1' and 'video_2' mappings
    """
    unit_ids = {'video_1': {}, 'video_2': {}}
    next_id = 0
    
    # Assign IDs to video 1 objects
    for obj_id in object_id_list_1.unique():
        unit_ids['video_1'][obj_id.item()] = next_id
        next_id += 1
    
    # Assign IDs to video 2 objects (reuse IDs for matched objects)
    for obj_id in object_id_list_2.unique():
        if object_sim_matrix.numel() == 0:
            max_sim = 0
        else:
            similarity_scores = object_sim_matrix[:, obj_id]
            max_sim = similarity_scores.max()
        
        if max_sim > args.object_similarity_threshold:
            # Matched object - use same ID from video 1
            matched_obj_id = object_id_list_1.unique()[similarity_scores.argmax()]
            unit_ids['video_2'][obj_id.item()] = unit_ids['video_1'][matched_obj_id.item()]
        else:
            # New object - assign new ID
            unit_ids['video_2'][obj_id.item()] = next_id
            next_id += 1
    
    return unit_ids


def save_object_masks(object_id_list, mask_indices, unit_ids, images, total_cluster_cost_maps,
                     img_length, H, W, colors, max_score, video_key, output_dir):
    """
    Save object masks and visualizations.
    
    Args:
        object_id_list: List of object IDs
        mask_indices: Pixel-to-object mapping
        unit_ids: Unified object ID mapping
        images: Input images
        total_cluster_cost_maps: Cost maps
        img_length: Number of images
        H, W: Image dimensions
        colors: Colormap for visualization
        max_score: Maximum cost for normalization
        video_key: 'video_1' or 'video_2'
        output_dir: Output directory
        
    Returns:
        Dictionary of object masks
    """
    object_masks = {}
    vis_dir = output_dir / f'{video_key}_detection'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for obj_idx, object_id in enumerate(object_id_list.unique()):
        object_id_indices = torch.where(object_id_list == object_id)[0]
        unified_id = unit_ids[video_key][object_id.item()]
        color = colors[unified_id % colors.shape[0]]
        
        if unified_id not in object_masks:
            object_masks[unified_id] = {}
        if video_key not in object_masks[unified_id]:
            object_masks[unified_id][video_key] = {}
        
        for single_idx in object_id_indices:
            single_mask = mask_indices == single_idx
            
            for frame_idx in range(img_length):
                frame_mask = single_mask.view(img_length, -1)[frame_idx].view(H, W)
                
                if not frame_mask.any():
                    continue
                
                # Merge with existing mask if present
                if frame_idx in object_masks[unified_id][video_key]:
                    old_mask = torch.tensor(mask_utils.decode(
                        object_masks[unified_id][video_key][frame_idx]['mask']
                    )).cuda()
                    frame_mask = frame_mask | old_mask
                
                # Convert to RLE
                rle = mask_to_rle_pytorch(frame_mask.unsqueeze(0))[0]
                rle = coco_encode_rle(rle)
                
                # Compute cost
                cost = total_cluster_cost_maps[frame_idx][frame_mask.bool()].mean().item()
                
                object_masks[unified_id][video_key][frame_idx] = {
                    'mask': rle,
                    'mask_size': frame_mask.sum().item(),
                    'cost': cost / max_score
                }
                
                # Save visualization
                img_offset = 0 if video_key == 'video_1' else img_length
                rgb_image = images[0][frame_idx + img_offset].permute(1, 2, 0).cpu().numpy()
                rgb_image[frame_mask.cpu().numpy().astype(bool)] = color.cpu().numpy()
                vis_path = vis_dir / f'{video_key}_frame_{frame_idx}_obj_{object_id}.png'
                plt.imsave(vis_path, rgb_image)
    
    return object_masks


def save_ground_truth_visualization(gt_file_path, images, img_1_length, H, W, 
                                    colors, output_dir, resample_rate):
    """Save ground truth object visualizations."""
    if not os.path.exists(gt_file_path):
        return
    
    gt = pickle.load(open(gt_file_path, 'rb'))
    gt_video_1_objects = gt['video1_objects']
    gt_video_2_objects = gt['video2_objects']
    gt_meta = gt.get('objects', {})
    
    for object_info in gt_meta:
        obj_k = object_info['original_obj_idx']
        in_video1 = object_info.get('in_video1', False)
        in_video2 = object_info.get('in_video2', False)
        
        # Process video 1
        if in_video1 and obj_k in gt_video_1_objects:
            for frame_idx, mask_rle in gt_video_1_objects[obj_k].items():
                frame_idx = int(frame_idx)
                if frame_idx % resample_rate != 0:
                    continue
                
                actual_idx = frame_idx // resample_rate
                gt_mask = torch.tensor(mask_utils.decode(mask_rle))
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W), mode='bilinear', align_corners=False
                )[0, 0] > 0.5
                
                rgb_image = images[0][actual_idx].permute(1, 2, 0).cpu().numpy()
                rgb_image[gt_mask.cpu().numpy()] = colors[obj_k].cpu().numpy()
                
                vis_dir = output_dir / 'video_1_detection'
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f'gt_video_1_frame_{actual_idx}_obj_{obj_k}.png'
                plt.imsave(vis_path, rgb_image)
        
        # Process video 2
        if in_video2 and obj_k in gt_video_2_objects:
            for frame_idx, mask_rle in gt_video_2_objects[obj_k].items():
                frame_idx = int(frame_idx)
                if frame_idx % resample_rate != 0:
                    continue
                
                actual_idx = frame_idx // resample_rate
                gt_mask = torch.tensor(mask_utils.decode(mask_rle))
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W), mode='bilinear', align_corners=False
                )[0, 0] > 0.5
                
                rgb_image = images[0][actual_idx + img_1_length].permute(1, 2, 0).cpu().numpy()
                rgb_image[gt_mask.cpu().numpy()] = colors[obj_k].cpu().numpy()
                
                vis_dir = output_dir / 'video_2_detection'
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis_path = vis_dir / f'gt_video_2_frame_{actual_idx}_obj_{obj_k}.png'
                plt.imsave(vis_path, rgb_image)


# ============================================================================
# Main Pipeline
# ============================================================================

def predict_changed_objects(file_list, img_1_length, img_2_length, output_dir, 
                visible_percentage=0.5, args=None, model=None):
    """
    Main pipeline for scene change detection.
    
    Args:
        file_list: List of image paths (video1 frames + video2 frames)
        img_1_length: Number of frames in video 1
        img_2_length: Number of frames in video 2
        output_dir: Output directory for results
        visible_percentage: Threshold for view similarity
        args: Configuration arguments
        model: Pi3 depth estimation model
    """
    start_time = time.time()
    scene_name = Path(file_list[0]).parents[1].stem
    
    print(f"\n{'='*80}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*80}\n")
    
    # ========== Stage 1: Load Images ==========
    stage_start = time.time()
    images = load_and_preprocess_images(file_list, DEVICE)
    H, W = images.shape[-2:]
    print(f"[Stage 1] Image loading: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 2: Depth and Pose Estimation ==========
    stage_start = time.time()
    res = estimate_depth_and_poses(images, model)
    intrinsic = compute_intrinsics(res, file_list, H, W)
    
    depth_map, point_map, poses, voxel_size = normalize_scene_scale(
        res['point_map'][0].cpu().numpy(),
        res['depth_map'],
        res['poses']
    )
    
    pi3_masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    pi3_masks = torch.logical_and(pi3_masks, non_edge)[0]
    
    print(f"[Stage 2] Depth and pose estimation: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 3: Similarity Matrix ==========
    stage_start = time.time()
    grid_intrinsic = torch.tensor([
        2.0 / W, 0, -1,
        0, 2.0 / H, -1
    ]).reshape(2, 3).to(DEVICE)
    img_coors = get_img_coor(H, W).to(DEVICE)
    
    similarity_matrix = compute_similarity_matrix(
        poses, depth_map, intrinsic, grid_intrinsic, img_coors,
        H, W, img_1_length, img_2_length
    )
    similarity_weights = compute_similarity_weights(similarity_matrix, visible_percentage)
    print(f"[Stage 3] Similarity matrix: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 4: Confidence Maps ==========
    stage_start = time.time()
    conf_list = compute_confidence_maps(images, model, file_list)
    del model
    torch.cuda.empty_cache()
    print(f"[Stage 4] Confidence computation: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 5: SAM Masks ==========
    stage_start = time.time()
    input_dir = Path(file_list[0]).parents[1]
    mask_file_path = input_dir / "sam_masks.pkl"
    masks_list = load_or_generate_sam_masks(file_list, H, W, mask_file_path)
    print(f"[Stage 5] SAM masks: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 6: Load DINOv3 ==========
    stage_start = time.time()
    dinov3_model = torch.hub.load(DINOV3_PATH, 'dinov3_vith16plus', 
                                  source='local').to(DEVICE).eval()
    print(f"[Stage 6] DINOv3 loading: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 7: Cost Map Computation ==========
    stage_start = time.time()
    file_output_dir = Path(output_dir) / scene_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for cached cost maps
    cache_filename = ('cost_maps.pkl', 'visible_maps.pkl')
    cached_cost_maps = load_cached_cost_maps(output_dir, scene_name, cache_filename, args)
    
    if cached_cost_maps is not None:
        total_cost_map_list, total_visible_map_list = cached_cost_maps
        print(f"[Stage 7] Cost maps loaded from cache: {time.time() - stage_start:.2f}s")
    else:
        total_cost_map_list = []
        total_visible_map_list = []
        
        for idx in range(len(file_list)):
            weighted_cost, visible_map = compute_cost_maps_for_image(
                idx, images, masks_list, poses, depth_map, intrinsic,
                conf_list, similarity_weights, pi3_masks, img_coors,
                grid_intrinsic, dinov3_model, H, W, args
            )
            total_cost_map_list.append(weighted_cost)
            total_visible_map_list.append(visible_map)
        
        if args.save_cache:
            # Save cost maps
            save_cost_maps(file_output_dir, cache_filename, 
                        total_cost_map_list, total_visible_map_list)
        print(f"[Stage 7] Cost map computation: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 8: Voxelization ==========
    stage_start = time.time()
    total_cost_map_list = torch.stack(total_cost_map_list)
    total_cost_maps = (args.lambda_geo * total_cost_map_list[:,0] + 
                      args.lambda_dino_region_match * total_cost_map_list[:,1] + 
                      args.lambda_dino * total_cost_map_list[:,2])
    total_cost_maps = total_cost_maps.to(DEVICE)
    total_visible_maps = torch.stack(total_visible_map_list, dim=0).view(-1).to(DEVICE)
    
    # Process each point cloud separately
    pc1_results = process_point_cloud_costs(
        point_map[:img_1_length], total_cost_maps[:img_1_length],
        total_visible_maps[:img_1_length * H * W], voxel_size, H, W
    )
    pc2_results = process_point_cloud_costs(
        point_map[img_1_length:], total_cost_maps[img_1_length:],
        total_visible_maps[img_1_length * H * W:], voxel_size, H, W
    )
    print(f"[Stage 8] Voxelization: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 9: DINOv3 Features ==========
    stage_start = time.time()
    pc1_dino_feat = extract_voxel_features(images[0][:img_1_length], 
                                           pc1_results['voxel_indices'],
                                           dinov3_model, H, W)
    pc2_dino_feat = extract_voxel_features(images[0][img_1_length:],
                                           pc2_results['voxel_indices'],
                                           dinov3_model, H, W)
    del dinov3_model
    torch.cuda.empty_cache()
    print(f"[Stage 9] DINOv3 feature extraction: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 10: Point Cloud Visualization ==========
    if args.vis_pc:
        stage_start = time.time()
        save_point_cloud_visualizations(
            point_map, pc1_results['cost_maps'], pc2_results['cost_maps'],
            images, img_1_length, file_output_dir
        )
        print(f"[Stage 10] Point cloud visualization: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 11: Object Detection ==========
    stage_start = time.time()
    
    # Compute cluster-level cost maps
    cluster_cost_1 = compute_cluster_costs(
        pc1_results['cost_maps'], masks_list[:img_1_length],
        total_visible_maps, img_1_length, H, W, args
    )
    cluster_cost_2 = compute_cluster_costs(
        pc2_results['cost_maps'], masks_list[img_1_length:],
        total_visible_maps, img_2_length, H, W, args, offset=img_1_length
    )
    
    # Detect objects
    object_list_1, mask_index_1, threshold_1 = detect_and_threshold_objects(
        cluster_cost_1, total_visible_maps[:img_1_length * H * W].view(img_1_length, H, W),
        masks_list[:img_1_length], pc1_results['voxel_locations'],
        pc1_results['voxel_indices'], pc1_dino_feat, img_1_length, H, W, args
    )
    object_list_2, mask_index_2, threshold_2 = detect_and_threshold_objects(
        cluster_cost_2, total_visible_maps[img_1_length * H * W:].view(img_2_length, H, W),
        masks_list[img_1_length:], pc2_results['voxel_locations'],
        pc2_results['voxel_indices'], pc2_dino_feat, img_2_length, H, W, args
    )
    
    # Merge objects across frames
    object_id_list_1, merged_object_list_1 = merge_objects(
        object_list_1,
        geometry_distance_threshold=args.geometry_distance_threshold_of_voxel_size * voxel_size,
        visual_threshold_ratio=args.visual_threshold_ratio,
        geometry_threshold_ratio=args.geometry_threshold_ratio,
        general_threshold=args.general_threshold
    )
    object_id_list_2, merged_object_list_2 = merge_objects(
        object_list_2,
        geometry_distance_threshold=args.geometry_distance_threshold_of_voxel_size * voxel_size,
        visual_threshold_ratio=args.visual_threshold_ratio,
        geometry_threshold_ratio=args.geometry_threshold_ratio,
        general_threshold=args.general_threshold
    )
    
    print(f"[Stage 11] Object detection: {time.time() - stage_start:.2f}s")
    print(f"  Detected {len(merged_object_list_1)} objects in video 1, "
          f"{len(merged_object_list_2)} objects in video 2")
    
    # ========== Stage 12: Object Association and Mask Generation ==========
    stage_start = time.time()
    
    # Compute object similarity
    object_sim_matrix = torch.zeros(len(merged_object_list_1), len(merged_object_list_2))
    for i, obj1 in enumerate(merged_object_list_1):
        for j, obj2 in enumerate(merged_object_list_2):
            object_sim_matrix[i, j] = torch.cosine_similarity(
                obj1['dino_feat'], obj2['dino_feat'], dim=0
            )
    
    # Create unified IDs
    unit_ids = create_unified_object_ids(
        object_id_list_1, object_id_list_2, object_sim_matrix, args
    )
    
    # Prepare colors for visualization
    colors = create_color_palette(DEVICE)
    
    # Save object masks
    object_masks = {'H': H, 'W': W}
    
    masks_1 = save_object_masks(
        object_id_list_1, mask_index_1, unit_ids, images, cluster_cost_1,
        img_1_length, H, W, colors, args.max_score, 'video_1', file_output_dir
    )
    masks_2 = save_object_masks(
        object_id_list_2, mask_index_2, unit_ids, images, cluster_cost_2,
        img_2_length, H, W, colors, args.max_score, 'video_2', file_output_dir
    )
    
    # Merge masks
    for obj_id, obj_data in masks_1.items():
        if obj_id not in object_masks:
            object_masks[obj_id] = {}
        object_masks[obj_id].update(obj_data)
    
    for obj_id, obj_data in masks_2.items():
        if obj_id not in object_masks:
            object_masks[obj_id] = {}
        object_masks[obj_id].update(obj_data)
    
    # Save object masks
    output_masks_file = file_output_dir / 'object_masks.pkl'
    with open(output_masks_file, 'wb') as f:
        pickle.dump(object_masks, f)
    
    # Save point cloud visualizations with object colors
    if args.vis_pc:
        save_object_point_clouds(
            point_map[:img_1_length].reshape(-1, 3),
            point_map[img_1_length:].reshape(-1, 3),
            object_id_list_1, object_id_list_2,
            mask_index_1, mask_index_2, unit_ids, colors,
            file_output_dir
        )
    
    print(f"[Stage 12] Object mask generation: {time.time() - stage_start:.2f}s")
    
    # ========== Stage 13: Ground Truth Visualization ==========
    stage_start = time.time()
    gt_file_path = Path(args.gt_dir) / scene_name / 'segments.pkl'
    save_ground_truth_visualization(
        gt_file_path, images, img_1_length, H, W, colors,
        file_output_dir, args.resample_rate
    )
    print(f"[Stage 13] Ground truth visualization: {time.time() - stage_start:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"{'='*80}\n")


# ============================================================================
# Helper Functions for Main Pipeline
# ============================================================================

def load_cached_cost_maps(output_dir, scene_name, cache_filenames, args):
    """Load cached cost maps if available."""
    if not args.use_cache:
        return None
    
    cost_filename, visible_filename = cache_filenames
    root_dir = Path(output_dir).parent
    
    for exp_dir in root_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        scene_dir = exp_dir / scene_name
        if not scene_dir.exists():
            continue
        
        cost_path = scene_dir / cost_filename
        visible_path = scene_dir / visible_filename
        
        if cost_path.exists() and visible_path.exists():
            with open(cost_path, 'rb') as f:
                cost_maps = pickle.load(f)
            with open(visible_path, 'rb') as f:
                visible_maps = pickle.load(f)
            print(f"Loaded cached cost maps from: {exp_dir.name}")
            return (cost_maps, visible_maps)
    
    return None


def save_cost_maps(output_dir, cache_filenames, cost_maps, visible_maps):
    """Save cost maps to disk."""
    cost_filename, visible_filename = cache_filenames
    
    with open(output_dir / cost_filename, 'wb') as f:
        pickle.dump(cost_maps, f)
    with open(output_dir / visible_filename, 'wb') as f:
        pickle.dump(visible_maps, f)


def process_point_cloud_costs(point_map, cost_maps, visible_masks, voxel_size, H, W):
    """Process costs for a point cloud through voxelization."""
    n_images = point_map.shape[0]
    points = point_map.reshape(-1, 3)
    
    # Voxelization
    point_to_voxel = voxelization(
        torch.tensor(points[None], device=DEVICE), voxel_size
    )[0]
    
    # Aggregate costs to voxels
    visible_points = visible_masks.view(-1)
    visible_costs = cost_maps.view(-1)[visible_points]
    visible_voxel_indices = point_to_voxel[visible_points]
    
    cost_per_voxel = scatter_mean(visible_costs, visible_voxel_indices, dim=0)
    
    # Create full cost map
    max_voxel_id = point_to_voxel.unique().max()
    full_cost_map = torch.zeros(max_voxel_id + 1, device=DEVICE)
    for voxel_id in visible_voxel_indices.unique():
        full_cost_map[voxel_id] = cost_per_voxel[voxel_id]
    
    point_costs = full_cost_map[point_to_voxel]
    
    # Compute voxel locations
    voxel_locations = scatter_mean(
        torch.tensor(points, device=DEVICE), point_to_voxel, dim=0
    )
    
    return {
        'cost_maps': point_costs,
        'voxel_indices': point_to_voxel,
        'voxel_locations': voxel_locations
    }


def extract_voxel_features(images, voxel_indices, dinov3_model, H, W):
    """Extract DINOv3 features and aggregate to voxels."""
    n_images = images.shape[0]
    feat_list = torch.zeros(n_images, H, W, APPEARANCE_FEATURE_DIM).to(DEVICE)
    
    for i in range(n_images):
        img_np = images[i].permute(1,2,0).cpu().numpy()
        feat = predict_dinov3_feat(img_np, dinov3_model)
        feat_list[i] = feat.permute(1, 2, 0)
    
    feat_flat = feat_list.view(-1, APPEARANCE_FEATURE_DIM)
    voxel_features = scatter_mean(feat_flat.cpu(), voxel_indices.cpu(), dim=0).cuda()
    
    return voxel_features


def compute_cluster_costs(cost_maps, masks_list, total_visible_maps, 
                         n_images, H, W, args, offset=0):
    """Aggregate voxel costs to SAM regions."""
    cluster_costs = []
    
    for idx in range(n_images):
        cluster_cost = torch.zeros(H, W).to(DEVICE)
        sam_mask = torch.tensor(masks_list[idx]).cuda()
        cost_map = cost_maps.view(-1, H, W)[idx].cuda()
        visible_map = total_visible_maps.view(-1, H, W)[idx + offset].cuda()
        
        for region_id in sam_mask.unique():
            if region_id == -1:
                continue
            
            region_mask = sam_mask == region_id
            
            valid_mask = region_mask & visible_map
            if (valid_mask.sum() / region_mask.sum()) > 0.3: 
                # Need to make sure the visible region is not too small compared to the whole region
                cluster_cost[region_mask] = cost_map[valid_mask].mean()
        
        cluster_costs.append(cluster_cost)
    
    return torch.stack(cluster_costs)


def create_color_palette(device):
    """Create a color palette for object visualization."""
    cmap1 = plt.cm.tab20
    cmap2 = plt.cm.tab20b
    cmap3 = plt.cm.tab20c
    
    colors1 = torch.tensor([cmap1(i)[:3] for i in range(20)]).float().to(device)
    colors2 = torch.tensor([cmap2(i)[:3] for i in range(20)]).float().to(device)
    colors3 = torch.tensor([cmap3(i)[:3] for i in range(20)]).float().to(device)
    
    return torch.cat([colors1, colors2, colors3], dim=0)


def save_point_cloud_visualizations(point_map, cost_1, cost_2, images, 
                                    img_1_length, output_dir):
    """Save point cloud visualizations with cost coloring."""
    # Combined visualization
    total_costs = torch.cat([cost_1, cost_2])
    cost_vis = apply_colormap(total_costs).reshape(len(point_map), -1, 3)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_map.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(cost_vis.reshape(-1, 3))
    o3d.io.write_point_cloud(str(output_dir / 'cost_map_merged.ply'), pcd)
    
    # Individual visualizations
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_map[:img_1_length].reshape(-1, 3))
    pcd1.colors = o3d.utility.Vector3dVector(
        apply_colormap(cost_1).reshape(-1, 3)
    )
    o3d.io.write_point_cloud(str(output_dir / 'cost_map_1.ply'), pcd1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_map[img_1_length:].reshape(-1, 3))
    pcd2.colors = o3d.utility.Vector3dVector(
        apply_colormap(cost_2).reshape(-1, 3)
    )
    o3d.io.write_point_cloud(str(output_dir / 'cost_map_2.ply'), pcd2)
    
    # RGB point clouds
    pcd_rgb1 = o3d.geometry.PointCloud()
    pcd_rgb1.points = o3d.utility.Vector3dVector(point_map[:img_1_length].reshape(-1, 3))
    pcd_rgb1.colors = o3d.utility.Vector3dVector(
        images[0][:img_1_length].permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
    )
    o3d.io.write_point_cloud(str(output_dir / 'rgb_pc_1.ply'), pcd_rgb1)
    
    pcd_rgb2 = o3d.geometry.PointCloud()
    pcd_rgb2.points = o3d.utility.Vector3dVector(point_map[img_1_length:].reshape(-1, 3))
    pcd_rgb2.colors = o3d.utility.Vector3dVector(
        images[0][img_1_length:].permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
    )
    o3d.io.write_point_cloud(str(output_dir / 'rgb_pc_2.ply'), pcd_rgb2)


def save_object_point_clouds(pc1_points, pc2_points, object_id_list_1, object_id_list_2,
                             mask_index_1, mask_index_2, unit_ids, colors, output_dir):
    """Save point clouds colored by object ID."""
    vis_pc_1 = torch.zeros(pc1_points.shape[0], 3).to(DEVICE)
    vis_pc_2 = torch.zeros(pc2_points.shape[0], 3).to(DEVICE)
    
    # Color video 1 objects
    for obj_id in object_id_list_1.unique():
        obj_indices = torch.where(object_id_list_1 == obj_id)[0]
        unified_id = unit_ids['video_1'][obj_id.item()]
        color = colors[unified_id % colors.shape[0]]
        
        for single_idx in obj_indices:
            mask = mask_index_1 == single_idx
            vis_pc_1[mask.view(-1)] = color
    
    # Color video 2 objects
    for obj_id in object_id_list_2.unique():
        obj_indices = torch.where(object_id_list_2 == obj_id)[0]
        unified_id = unit_ids['video_2'][obj_id.item()]
        color = colors[unified_id % colors.shape[0]]
        
        for single_idx in obj_indices:
            mask = mask_index_2 == single_idx
            vis_pc_2[mask.view(-1)] = color
    
    # Save point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1_points)
    pcd1.colors = o3d.utility.Vector3dVector(vis_pc_1.cpu().numpy())
    o3d.io.write_point_cloud(str(output_dir / 'object_instances_1.ply'), pcd1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2_points)
    pcd2.colors = o3d.utility.Vector3dVector(vis_pc_2.cpu().numpy())
    o3d.io.write_point_cloud(str(output_dir / 'object_instances_2.ply'), pcd2)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SceneDiff: Multi-view Scene Change Detection'
    )
    
    # I/O arguments
    parser.add_argument("--video_1_dir", nargs='+', default=['derek_working_room_1'])
    parser.add_argument("--video_2_dir", nargs='+', default=['derek_working_room_2'])
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for results")
    parser.add_argument('--gt_dir', type=str, 
                       default='/work/nvme/bcqn/ywu20/scene_change/final_data/results_valid/',
                       help='Directory containing ground truth data')
    
    # Processing arguments
    parser.add_argument("--resample_rate", type=int, default=30,
                       help="Sample 1 frame every N frames")
    parser.add_argument("--visible_percentage", type=float, default=0.5,
                       help="Minimum overlap for view similarity")
    
    # Cost computation arguments
    parser.add_argument("--lambda_geo", type=float, default=1.0,
                       help="Weight for geometric cost")
    parser.add_argument("--lambda_dino", type=float, default=0.3,
                       help="Weight for DINOv3 rendering cost")
    parser.add_argument("--lambda_dino_region_match", type=float, default=0.1,
                       help="Weight for DINOv3 region matching cost")
    parser.add_argument("--occlusion_threshold", type=float, default=-0.02,
                       help="Threshold for occlusion detection")
    
    # Object detection arguments
    parser.add_argument("--min_detection_pixel", type=int, default=100,
                       help="Minimum pixels for object detection")
    parser.add_argument("--threshold_method", type=str, default='otsu',
                       choices=['otsu', 'max_entropy'],
                       help="Method for cost thresholding")
    parser.add_argument("--change_region_threshold", type=float, default=None,
                       help="Manual threshold (overrides auto threshold)")
    parser.add_argument("--filter_percentage_before_threshold", type=float, default=0.6,
                       help="Filter low costs before thresholding")
    parser.add_argument("--max_score", type=float, default=0.5,
                       help="Maximum cost for normalization")
    parser.add_argument("--max_detected_objects", type=int, default=1000,
                       help="Maximum number of detected objects")
    
    # Object merging arguments
    parser.add_argument("--geometry_distance_threshold_of_voxel_size", type=int, default=2,
                       help="Geometry distance threshold in voxel units")
    parser.add_argument("--visual_threshold_ratio", type=float, default=0.7,
                       help="Visual similarity threshold for merging")
    parser.add_argument("--geometry_threshold_ratio", type=float, default=0.5,
                       help="Geometry threshold for merging")
    parser.add_argument("--general_threshold", type=float, default=1.4,
                       help="General threshold for merging")
    parser.add_argument("--object_similarity_threshold", type=float, default=0.7,
                       help="Threshold for object association across videos")
    
    # Aggregation arguments
    parser.add_argument("--mean_over_visible_region_during_cost_aggregation", 
                       type=eval, default=True,
                       help="Use only visible regions for cost aggregation")
    parser.add_argument("--merge_pc_before_and_after", type=eval, default=True,
                       help="Merge point clouds before thresholding")
    
    # Dataset arguments
    parser.add_argument("--splits", type=str, default='all',
                       choices=['val', 'test', 'all'],
                       help="Dataset split to process")
    parser.add_argument("--sets", type=str, default='All',
                       choices=['Diverse', 'Kitchen', 'All'],
                       help="Dataset subset to process")
    
    # Miscellaneous
    parser.add_argument("--use_cache", type=eval, default=False,
                       help="Use cached cost maps if available")
    parser.add_argument("--save_cache", type=eval, default=False,
                       help="Save cache of cost maps")
    parser.add_argument("--vis_pc", type=eval, default=False,
                       help="Save point cloud visualizations")
    
    args = parser.parse_args()
    
    print(f"Starting job: {args.output_dir}")
    
    # Initialize model
    model = Pi3.from_pretrained("yyfz233/Pi3").to(DEVICE).eval()
    
    # Load dataset splits
    val_split = json.load(open('splits/val_split.json', 'r'))
    val_split['All'] = val_split['Diverse'] + val_split['Kitchen']
    
    test_split = json.load(open('splits/test_split.json', 'r'))
    test_split['All'] = test_split['Diverse'] + test_split['Kitchen']
    
    # Determine scenes to process
    if args.splits == 'val':
        valid_scenes = val_split[args.sets]
    elif args.splits == 'test':
        valid_scenes = test_split[args.sets]
    elif args.splits == 'all':
        valid_scenes = val_split[args.sets] + test_split[args.sets]
    else:
        raise ValueError(f"Invalid split: {args.splits}")
    
    print(f"Processing: {args.splits} split, {args.sets} set")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter already processed scenes
    all_scenes = sorted(Path(args.gt_dir).iterdir())
    finished_scenes = []
    
    for scene_dir in all_scenes:
        for output_subdir in output_dir.glob(f'{scene_dir.name}*'):
            if output_subdir.is_dir() and (output_subdir / 'object_masks.pkl').exists():
                print(f'{output_subdir} already processed')
                try:
                    valid_scenes.remove(output_subdir.name)
                    finished_scenes.append(output_subdir.name)
                except ValueError:
                    pass
                break
    
    print(f'Scenes to process: {len(valid_scenes)}')
    print(f'Already processed: {len(finished_scenes)}')
    
    # Process each scene
    for scene_name in valid_scenes:
        scene_dir = Path(args.gt_dir) / scene_name
        if not scene_dir.exists():
            continue
        
        video_1_dir = scene_dir / 'video1_frames'
        video_2_dir = scene_dir / 'video2_frames'
        
        # Extract frames from videos if needed
        video_files = sorted([
            x for x in os.listdir(scene_dir)
            if x.lower().endswith(('.mp4', '.mov', '.avi'))
        ])
        
        if len(video_files) >= 2:
            video_1_path = scene_dir / video_files[0]
            video_2_path = scene_dir / video_files[1]
            
            if not video_1_dir.exists() or not video_2_dir.exists():
                process_video_to_frames(str(video_1_path), str(video_1_dir))
                process_video_to_frames(str(video_2_path), str(video_2_dir))
        
        # Load frame lists
        img_1_list = sorted(video_1_dir.glob('*.jpg'))[::args.resample_rate]
        img_2_list = sorted(video_2_dir.glob('*.jpg'))[::args.resample_rate]
        
        if len(img_1_list) == 0 or len(img_2_list) == 0:
            print(f"Skipping {scene_name}: no frames found")
            continue
        
        file_list = img_1_list + img_2_list
        
        print(f"\nProcessing scene: {scene_name}")
        print(f"  Video 1: {len(img_1_list)} frames")
        print(f"  Video 2: {len(img_2_list)} frames")
        
        global img_1_static_list, img_2_static_list
        img_1_static_list = img_1_list
        img_2_static_list = img_2_list
        
        try:
            predict_changed_objects(
                file_list, len(img_1_list), len(img_2_list),
                args.output_dir, args.visible_percentage, args, model
            )
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
