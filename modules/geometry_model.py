"""
Geometry Model Module
=====================

Handles depth estimation, camera pose computation, and geometric cost calculations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from utils import (
    recover_focal_shift,
    get_robust_voxel_size,
    get_img_coor,
    valid_mask_after_proj,
    foreground_distance,
    compute_conf_norm_distance,
)


class GeometryModel:
    """
    Handles all geometry-related computations including depth estimation,
    pose estimation, and geometric cost computation.
    """
    
    def __init__(self, config, device='cuda'):
        """
        Initialize geometry model.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.model = None


    def _load_pi3_model(self):
        """Load Pi3 depth estimation model."""
        import sys
        import os
        # Get project root (go up from modules/ to project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pi3_path = os.path.join(project_root, 'submodules/Pi3')
        if pi3_path not in sys.path:
            sys.path.append(pi3_path)
        
        from pi3.models.pi3 import Pi3
        from pi3.utils.geometry import depth_edge
        
        self.depth_edge_fn = depth_edge
        
        model_name = self.config['models']['pi3']['name']
        self.model = Pi3.from_pretrained(model_name).to(self.device).eval()
        print(f"[GeometryModel] Loaded Pi3 model: {model_name}")
    
    def estimate_depth_and_poses(self, images):
        """
        Estimate depth maps and camera poses for input images.
        
        Args:
            images: Input images (B, N, C, H, W)
            
        Returns:
            Dictionary containing depth, poses, intrinsics, and point maps
        """
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = self.model(images)
        
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
    
    def compute_intrinsics(self, res, file_list, H, W):
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
    
    def normalize_scene_scale(self, point_map, depth_map, poses):
        """
        Normalize scene scale using voxel size estimation.
        
        Args:
            point_map: Point cloud (N, H, W, 3)
            depth_map: Depth maps (B, N, H, W, 1)
            poses: Camera poses (B, N, 4, 4)
            
        Returns:
            Tuple of (scaled_depth, scaled_points, scaled_poses, voxel_size)
        """
        subsample_size = self.config['processing']['subsample_size']
        total_voxel_number = self.config['processing']['total_voxel_number']
        
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
    
    def compute_per_image_confidence(self, single_image, threshold=0.5):
        """
        Compute confidence map for a single image.
        
        Args:
            single_image: Input image tensor (C, H, W)
            threshold: Confidence threshold percentile
            
        Returns:
            Confidence map (H, W)
        """
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = self.model(single_image[None])
        
        depth_conf = torch.sigmoid(res['conf'][..., 0])
        
        conf_max = torch.kthvalue(
            depth_conf.reshape(-1),
            int(threshold * depth_conf.reshape(-1).numel())
        ).values.item()
        
        H, W = depth_conf.shape[-2:]
        conf = (depth_conf / conf_max).clamp(0, 1).view(H, W)
        
        return conf
    
    def compute_geometric_costs(self, pose_1, depth_1, intrinsic_1, conf_1,
                                pose_2, depth_2, intrinsic_2, conf_2,
                                img_coors, grid_intrinsic, H, W):
        """
        Compute geometric costs between two views.
        
        Args:
            pose_1, depth_1, intrinsic_1, conf_1: First view parameters
            pose_2, depth_2, intrinsic_2, conf_2: Second view parameters
            img_coors: Image coordinates
            grid_intrinsic: Grid intrinsic matrix
            H, W: Image dimensions
            
        Returns:
            Dictionary with visibility masks and geometric costs
        """
        # Compute visibility
        vis_1 = valid_mask_after_proj(
            intrinsic_1.cuda(), pose_1.cuda(), depth_1,
            intrinsic_2.cuda(), pose_2.cuda(), depth_2,
            img_coors, H, W
        )
        vis_2 = valid_mask_after_proj(
            intrinsic_2.cuda(), pose_2.cuda(), depth_2,
            intrinsic_1.cuda(), pose_1.cuda(), depth_1,
            img_coors, H, W
        )
        
        # Compute foreground distances
        foreground_dis_1 = foreground_distance(
            intrinsic_1.cuda(), pose_1.cuda(), depth_1,
            intrinsic_2.cuda(), pose_2.cuda(), depth_2,
            img_coors, grid_intrinsic, H, W
        )
        foreground_dis_2 = foreground_distance(
            intrinsic_2.cuda(), pose_2.cuda(), depth_2,
            intrinsic_1.cuda(), pose_1.cuda(), depth_1,
            img_coors, grid_intrinsic, H, W
        )
        
        # Compute geometric costs
        fg_pixel_geo_conf_1 = compute_conf_norm_distance(foreground_dis_1, conf_1)
        fg_pixel_geo_conf_2 = compute_conf_norm_distance(foreground_dis_2, conf_2)
        
        return {
            'vis_1': vis_1,
            'vis_2': vis_2,
            'foreground_dis_1': foreground_dis_1,
            'foreground_dis_2': foreground_dis_2,
            'geo_cost_1': fg_pixel_geo_conf_1,
            'geo_cost_2': fg_pixel_geo_conf_2
        }
    
    def get_valid_depth_masks(self, res):
        """
        Get valid depth masks from model output.
        
        Args:
            res: Model output dictionary
            
        Returns:
            Valid depth masks
        """
        pi3_masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
        non_edge = ~self.depth_edge_fn(res['local_points'][..., 2], rtol=0.03)
        pi3_masks = torch.logical_and(pi3_masks, non_edge)[0]
        
        return pi3_masks
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
