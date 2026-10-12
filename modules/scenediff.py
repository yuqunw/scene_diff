"""
SceneDiff Main Pipeline
=======================

Main orchestrator class for multi-view scene change detection.
"""

import os
import time
import pickle
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from pycocotools import mask as mask_utils
from skimage.filters import threshold_otsu
from torch_scatter import scatter_mean

from .geometry_model import GeometryModel
from .mask_model import MaskModel
from .semantic_model import SemanticModel

from utils import (
    load_images_as_tensor_from_list,
    voxelization,
    get_img_coor,
    calculate_mask_percentage,
    apply_colormap,
    threshold_maximum_entropy,
    mask_to_rle_pytorch,
    coco_encode_rle,
    merge_objects,
    reprojected_feature,
    get_dino_matched_region_cost,
)


class SceneDiff:
    """
    Main class for multi-view scene change detection.
    
    Orchestrates geometry, mask, and semantic models to detect and track
    changed objects across video sequences.
    """
    
    def __init__(self, config, device='cuda'):
        """
        Initialize SceneDiff pipeline.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        
        # Initialize sub-models
        print("Initializing SceneDiff pipeline...")
        self.geometry_model = GeometryModel(config, device)
        self.mask_model = MaskModel(config, device)
        self.semantic_model = SemanticModel(config, device)
        self.feature_dim = config['models']['dinov3']['feature_dim']
        
        print("SceneDiff pipeline initialized successfully!")
    
    def process_scene(self, file_list, img_1_length, img_2_length, output_dir, num_closest_views=3):
        """
        Main pipeline for scene change detection.
        
        Args:
            file_list: List of image paths (video1 frames + video2 frames)
            img_1_length: Number of frames in video 1
            img_2_length: Number of frames in video 2
            output_dir: Output directory for results
            num_closest_views: Number of closest cross-video views (by pose distance)
                to use for each target view
            
        Returns:
            Dictionary containing detected objects and masks
        """
        start_time = time.time()
        scene_name = Path(file_list[0]).parents[1].stem
        
        print(f"\n{'='*80}")
        print(f"Processing scene: {scene_name}")
        print(f"{'='*80}\n")
        
        # Stage 1: Load Images
        stage_start = time.time()
        max_size = int(self.config.get('processing', {}).get('max_size', 518))
        images = load_images_as_tensor_from_list(file_list, max_size=max_size)[None].to(self.device)
        H, W = images.shape[-2:]
        print(f"[Stage 1] Image loading: {time.time() - stage_start:.2f}s")
        
        # Stage 2: Depth and Pose Estimation
        stage_start = time.time()
        self.geometry_model._load_pi3_model() # Load the model when needed
        res = self.geometry_model.estimate_depth_and_poses(images)
        intrinsic = self.geometry_model.compute_intrinsics(res, file_list, H, W)
        
        depth_map, point_map, poses, voxel_size = self.geometry_model.normalize_scene_scale(
            res['point_map'][0].cpu().numpy(),
            res['depth_map'],
            res['poses']
        )
        
        pi3_masks = self.geometry_model.get_valid_depth_masks(res)
        print(f"[Stage 2] Depth and pose estimation: {time.time() - stage_start:.2f}s")
        
        # Stage 3: Similarity Matrix
        stage_start = time.time()
        grid_intrinsic = torch.tensor([
            2.0 / W, 0, -1,
            0, 2.0 / H, -1
        ]).reshape(2, 3).to(self.device)
        img_coors = get_img_coor(H, W).to(self.device)
        
        similarity_matrix = self._compute_similarity_matrix(
            poses, depth_map, intrinsic, grid_intrinsic, img_coors,
            H, W, img_1_length, img_2_length
        )
        similarity_weights = self._compute_similarity_weights(similarity_matrix)
        print(f"[Stage 3] Similarity matrix: {time.time() - stage_start:.2f}s")
        
        # Stage 4: Confidence Maps
        stage_start = time.time()
        conf_list = self._compute_confidence_maps(images, file_list)
        print(f"[Stage 4] Confidence computation: {time.time() - stage_start:.2f}s")
        self.geometry_model.cleanup() # Clean up the model to free up memory

        # Stage 5: SAM Masks
        stage_start = time.time()
        input_dir = Path(file_list[0]).parents[1]
        mask_file_path = input_dir / "sam_masks.pkl"
        masks_list = self.mask_model.load_or_generate_masks(file_list, H, W, mask_file_path)
        print(f"[Stage 5] SAM masks: {time.time() - stage_start:.2f}s")
        self.mask_model.cleanup() # Clean up the mask model to free up memory

        output_dir = Path(output_dir)
        # Avoid duplicate nesting when caller already provides a scene-specific path.
        file_output_dir = output_dir if output_dir.name == scene_name else output_dir / scene_name
        file_output_dir.mkdir(parents=True, exist_ok=True)
        rand_colors = np.random.rand(1000, 3)
        for mask_index, single_mask in enumerate(masks_list):
            single_mask = cv2.resize(single_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            single_mask_vis = np.zeros((H, W, 3))
            for i in np.unique(single_mask):
                if i == -1:
                    continue
                mask_i = single_mask == i
                single_mask_vis[mask_i] = rand_colors[i]
            plt.imsave(f'{file_output_dir}/single_mask_vis_{mask_index}.png', single_mask_vis)

        # Stage 6: Cost Map Computation
        stage_start = time.time()
        
        self.semantic_model._load_dinov3_model() # Load the model when needed
        total_cost_maps, total_visible_maps = self._compute_cost_maps(
            images, masks_list, poses, depth_map, intrinsic, conf_list,
            similarity_weights, pi3_masks, img_coors, grid_intrinsic,
            H, W, file_output_dir, scene_name, output_dir,
            img_1_length, img_2_length, num_closest_views, file_list
        )
        # Ignore points outside normalized scene cube in all downstream visibility-driven steps.
        in_cube_mask = self._get_in_cube_point_mask(point_map).to(self.device)
        if in_cube_mask is not None:
            total_visible_maps = total_visible_maps & in_cube_mask
        print(f"[Stage 6] Cost map computation: {time.time() - stage_start:.2f}s")
        
        # Stage 7: Voxelization and Feature Extraction
        stage_start = time.time()
        pc1_results, pc2_results, pc1_dino_feat, pc2_dino_feat = self._voxelize_and_extract_features(
            point_map, total_cost_maps, total_visible_maps, images,
            img_1_length, img_2_length, voxel_size, H, W
        )
        print(f"[Stage 7] Voxelization and features: {time.time() - stage_start:.2f}s")
        
        # Stage 8: Object Detection and Merging
        stage_start = time.time()
        cluster_cost_1, cluster_cost_2 = self._compute_cluster_costs(
            pc1_results, pc2_results, masks_list, total_visible_maps,
            img_1_length, img_2_length, H, W
        )
        
        object_results = self._detect_and_merge_objects(
            cluster_cost_1, cluster_cost_2, total_visible_maps,
            masks_list, pc1_results, pc2_results, pc1_dino_feat, pc2_dino_feat,
            img_1_length, img_2_length, H, W, voxel_size
        )
        print(f"[Stage 8] Object detection: {time.time() - stage_start:.2f}s")
        print(f"  Detected {len(object_results['merged_objects_1'])} objects in video 1, "
              f"{len(object_results['merged_objects_2'])} objects in video 2")
        self.semantic_model.cleanup() # Clean up the semantic model to free up memory

        # Stage 9: Object Association and Mask Generation
        stage_start = time.time()
        object_masks, unit_ids = self._associate_and_save_objects(
            object_results, images, cluster_cost_1, cluster_cost_2,
            img_1_length, img_2_length, H, W, file_output_dir
        )
        self._save_video2_difference_masks(
            object_masks=object_masks,
            unit_ids=unit_ids,
            poses=poses[0],
            intrinsics=intrinsic[0],
            point_map=point_map,
            img_1_length=img_1_length,
            img_2_length=img_2_length,
            H=H,
            W=W,
            output_dir=file_output_dir
        )
        print(f"[Stage 9] Object mask generation: {time.time() - stage_start:.2f}s")
        
        # Stage 10: Visualization (if enabled)
        if self.config['visualization']['vis_pc']:
            self._save_visualizations(
                point_map, pc1_results, pc2_results, images, object_results, unit_ids,
                img_1_length, file_output_dir
            )
        
        # Stage 11: Ground Truth Visualization (if available)
        if self.config['visualization']['save_gt_visualization']:
            self._save_ground_truth_visualization(
                scene_name, images, img_1_length, H, W, file_output_dir
            )
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"{'='*80}\n")
        
        return object_masks
    
    def _compute_similarity_matrix(self, poses, depth_map, intrinsic, grid_intrinsic,
                                   img_coors, H, W, img_1_length, img_2_length):
        """Compute pairwise similarity between images from two sets."""
        n_total = img_1_length + img_2_length
        similarity_matrix = -1 * torch.ones(n_total, n_total).cuda()
        
        for i in range(img_1_length):
            for j in range(img_2_length):
                set_2_idx = img_1_length + j
                
                sim_1 = calculate_mask_percentage(
                    poses[0][i], poses[0][set_2_idx],
                    depth_map[0][i].permute(2, 0, 1), depth_map[0][set_2_idx].permute(2, 0, 1),
                    intrinsic[0][i], intrinsic[0][set_2_idx],
                    grid_intrinsic, 0.3, img_coors, H, W
                )
                
                sim_2 = calculate_mask_percentage(
                    poses[0][set_2_idx], poses[0][i],
                    depth_map[0][set_2_idx].permute(2, 0, 1), depth_map[0][i].permute(2, 0, 1),
                    intrinsic[0][set_2_idx], intrinsic[0][i],
                    grid_intrinsic, 0.3, img_coors, H, W
                )
                
                similarity_matrix[i, set_2_idx] = (sim_1 + sim_2) / 2
                similarity_matrix[set_2_idx, i] = similarity_matrix[i, set_2_idx]
        
        return similarity_matrix
    
    def _compute_similarity_weights(self, similarity_matrix):
        """Convert similarity matrix to weights for aggregation. 
        For frames without views in another sequence with more than 50% similarity, set the highest similarity weight to 1."""
        percentage_threshold = self.config['processing']['visible_percentage']
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

    
    def compute_rendering_cost(self, feat_src, feat_tgt,
                              intrinsic_src, pose_src, depth_src,
                              intrinsic_tgt, pose_tgt,
                              img_coors, grid_intrinsic, H, W,
                              occlusion_mask, sam_mask, vis_mask):
        """
        Compute rendering-based appearance cost.
        
        Args:
            feat_src, feat_tgt: Source and target features
            intrinsic_src, pose_src, depth_src: Source camera parameters
            intrinsic_tgt, pose_tgt: Target camera parameters
            img_coors: Image coordinates
            grid_intrinsic: Grid intrinsic matrix
            H, W: Image dimensions
            occlusion_mask: Occlusion mask
            sam_mask: Segmentation mask
            vis_mask: Visibility mask
            
        Returns:
            Region-aggregated rendering cost (H, W)
        """
        # Reproject features
        feat_reprojected = reprojected_feature(
            intrinsic_src.cuda(), pose_src.cuda(), depth_src.view(1, H, W),
            intrinsic_tgt.cuda(), pose_tgt.cuda(), feat_tgt,
            img_coors, grid_intrinsic, H, W
        )
        
        # Compute cosine distance
        diff = 1 - torch.cosine_similarity(feat_reprojected, feat_src, dim=0)
        diff[occlusion_mask] = 0

        # # Visualize features using PCA and plt.imshow
        # from sklearn.decomposition import PCA
        # import numpy as np

        # # Get features as (C, H*W)
        # feat_reprojected_flat = feat_reprojected.view(feat_reprojected.shape[0], -1).permute(1, 0).cpu().detach().numpy()  # shape (H*W, C)
        # feat_src_flat = feat_src.view(feat_src.shape[0], -1).permute(1, 0).cpu().detach().numpy()  # shape (H*W, C)

        # # Fit PCA on concatenated features (to align spaces)
        # X = np.concatenate([feat_reprojected_flat, feat_src_flat], axis=0)
        # pca = PCA(n_components=3)
        # X_pca = pca.fit_transform(X)
        # feat_reprojected_pca = X_pca[:feat_reprojected_flat.shape[0]].reshape(H, W, 3)
        # feat_src_pca = X_pca[feat_reprojected_flat.shape[0]:].reshape(H, W, 3)

        # # Normalize for visualization
        # feat_reprojected_img = (feat_reprojected_pca - feat_reprojected_pca.min()) / (feat_reprojected_pca.max() - feat_reprojected_pca.min())
        # feat_src_img = (feat_src_pca - feat_src_pca.min()) / (feat_src_pca.max() - feat_src_pca.min())

        # plt.figure(figsize=(10,5))
        # plt.subplot(1,2,1)
        # plt.title("feat_reprojected PCA")
        # plt.imshow(feat_reprojected_img)
        # plt.axis('off')
        # plt.subplot(1,2,2)
        # plt.title("feat_src PCA")
        # plt.imshow(feat_src_img)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        
        # Aggregate to regions
        region_diff = torch.zeros_like(sam_mask).float()
        for region_id in sam_mask.unique():
            if region_id == -1:
                continue
            region_mask = sam_mask == region_id
            valid_mask = region_mask & ~occlusion_mask
            
            if valid_mask.sum() > 0:
                region_diff[region_mask] = diff[valid_mask].mean()
        
        return region_diff
    
    def aggregate_features_to_voxels(self, images, voxel_indices, H, W):
        """
        Extract features and aggregate to voxels.
        
        Args:
            images: Input images (N, C, H, W)
            voxel_indices: Point-to-voxel mapping (N*H*W,)
            H, W: Image dimensions
            
        Returns:
            Voxel-aggregated features (N_voxels, feature_dim)
        """
        n_images = images.shape[0]
        voxel_indices = voxel_indices.view(n_images, H * W).long().cpu()
        num_voxels = int(voxel_indices.max().item()) + 1

        # Online accumulation to avoid materializing all per-frame features at once.
        voxel_feat_sum = torch.zeros(num_voxels, self.feature_dim, dtype=torch.float32)
        voxel_feat_count = torch.zeros(num_voxels, dtype=torch.float32)

        for i in range(n_images):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            feat = self.semantic_model.extract_features(img_np)
            feat_flat = feat.permute(1, 2, 0).reshape(-1, self.feature_dim).detach().cpu().float()

            frame_voxel_idx = voxel_indices[i]
            voxel_feat_sum.index_add_(0, frame_voxel_idx, feat_flat)
            voxel_feat_count.index_add_(
                0,
                frame_voxel_idx,
                torch.ones(frame_voxel_idx.shape[0], dtype=torch.float32),
            )

        voxel_features = voxel_feat_sum / voxel_feat_count.clamp_min(1e-6).unsqueeze(-1)
        return voxel_features.to(self.device)

    def _compute_confidence_maps(self, images, file_list):
        """Compute confidence maps for all images."""
        conf_list = []
        for i in range(len(file_list)):
            conf = self.geometry_model.compute_per_image_confidence(images[0][i:i+1].clone())
            conf_list.append([conf])
        return conf_list
    
    def _compute_cost_maps(self, images, masks_list, poses, depth_map, intrinsic,
                          conf_list, similarity_weights, pi3_masks, img_coors,
                          grid_intrinsic, H, W, file_output_dir, scene_name, output_dir,
                          img_1_length, img_2_length, num_closest_views=3, file_list=None):
        """Compute cost maps for all images."""
        # Check for cache
        cache_config = self.config['cache']
        cached = None
        scene_group = Path(file_list[0]).parents[2].name if (file_list is not None and len(Path(file_list[0]).parents) >= 3) else None
        if cache_config['use_cache']:
            cached = self._load_cached_cost_maps(file_output_dir, scene_name, scene_group=scene_group)
            if cached is not None:
                total_cost_map_list, total_visible_map_list = cached
        
        if (cached is None) or (not cache_config['use_cache']):
            # Compute costs
            total_cost_map_list = []
            total_visible_map_list = []
            
            for idx in range(len(masks_list)):
                weighted_cost, visible_map = self._compute_cost_maps_for_image(
                    idx, images, masks_list, poses, depth_map, intrinsic,
                    conf_list, similarity_weights, pi3_masks, img_coors,
                    grid_intrinsic, H, W, img_1_length, img_2_length,
                    num_closest_views, file_list
                )
                total_cost_map_list.append(weighted_cost)
                total_visible_map_list.append(visible_map)
            
            # Save cache if enabled
            if cache_config['save_cache']:
                self._save_cost_maps(file_output_dir, total_cost_map_list, total_visible_map_list)
        
        # Aggregate costs
        total_cost_map_list = torch.stack(total_cost_map_list)
        costs_config = self.config['costs']
        total_cost_maps = (
            costs_config['lambda_geo'] * total_cost_map_list[:, 0] +
            costs_config['lambda_dino_region_match'] * total_cost_map_list[:, 1] +
            costs_config['lambda_dino'] * total_cost_map_list[:, 2]
        )
        total_cost_maps = total_cost_maps.to(self.device)
        total_visible_maps = torch.stack(total_visible_map_list, dim=0).to(self.device)
        
        return total_cost_maps, total_visible_maps
    
    def _compute_cost_maps_for_image(self, index, images, masks_list, poses, depth_map,
                                    intrinsic, conf_list, similarity_weights, pi3_masks,
                                    img_coors, grid_intrinsic, H, W,
                                    img_1_length, img_2_length, num_closest_views=3,
                                    file_list=None):
        """Compute cost maps for a single image."""
        cost_map_list = []
        visible_map_list = []
        
        sam_mask_1 = torch.tensor(masks_list[index]).cuda()
        pose_1 = poses[0][index].clone()
        depth_1 = depth_map[0][index].view(1, H, W).clone()
        intrinsic_1 = intrinsic[0][index].clone()
        conf_1 = conf_list[index][0]
        
        # Extract features once
        dinov3_feat_1 = self._extract_semantic_feature_by_path(
            index, images, file_list
        )
        
        closest_indices = self._select_pose_closest_cross_video_views(
            index, poses, similarity_weights, img_1_length, img_2_length, num_closest_views
        )
        for closest_index in closest_indices:
            
            sam_mask_2 = torch.tensor(masks_list[closest_index]).cuda()
            pose_2 = poses[0][closest_index].clone()
            depth_2 = depth_map[0][closest_index].view(1, H, W).clone()
            intrinsic_2 = intrinsic[0][closest_index].clone()
            conf_2 = conf_list[closest_index][0]
            
            # Compute geometric costs
            geo_results = self.geometry_model.compute_geometric_costs(
                pose_1, depth_1, intrinsic_1, conf_1,
                pose_2, depth_2, intrinsic_2, conf_2,
                img_coors, grid_intrinsic, H, W
            )
            
            # Aggregate geometric costs to regions
            fg_region_geo_cost_1 = self.mask_model.aggregate_to_regions(
                geo_results['geo_cost_1'] * geo_results['vis_1'], sam_mask_1
            )
            
            # Compute semantic costs
            dinov3_feat_2 = self._extract_semantic_feature_by_path(
                closest_index, images, file_list
            )
            
            occlusion_threshold = self.config['processing']['occlusion_threshold']
            occlusion_mask_1 = (
                (geo_results['foreground_dis_1'] < occlusion_threshold) |
                ~pi3_masks[index] | ~geo_results['vis_1']
            )
            occlusion_mask_2 = (
                (geo_results['foreground_dis_2'] < occlusion_threshold) |
                ~pi3_masks[closest_index] | ~geo_results['vis_2']
            )
            
            # DINOv3 region matching cost
            dino_region_match_cost_1, _, _, _, _, _ = get_dino_matched_region_cost(
                dinov3_feat_1, dinov3_feat_2, sam_mask_1, sam_mask_2,
                ~occlusion_mask_1, ~occlusion_mask_2,
                occlusion_threshold=self.config['costs'].get('region_match_occlusion_threshold', 0.5)
            )
            
            # DINOv3 rendering cost
            dino_render_cost_1 = self.compute_rendering_cost(
                dinov3_feat_1, dinov3_feat_2, intrinsic_1, pose_1, depth_1,
                intrinsic_2, pose_2, img_coors, grid_intrinsic, H, W,
                occlusion_mask_1, sam_mask_1, geo_results['vis_1']
            )
            
            # Stack costs
            costs = torch.stack([
                (fg_region_geo_cost_1 * geo_results['vis_1']).cpu(),
                (dino_region_match_cost_1 * geo_results['vis_1']).cpu(),
                (dino_render_cost_1 * geo_results['vis_1']).cpu()
            ], dim=0)
            
            cost_map_list.append(costs)
            visible_map_list.append((~occlusion_mask_1).cpu())

        if len(cost_map_list) == 0:
            return torch.zeros((3, H, W)), torch.zeros((H, W), dtype=torch.bool)

        # Aggregate costs across views
        cost_maps = torch.stack(cost_map_list)
        visible_maps = torch.stack(visible_map_list)
        
        visible_count = visible_maps.sum(dim=0)
        weighted_cost = (cost_maps * visible_maps[:, None, :, :]).sum(dim=0)
        weighted_cost = weighted_cost / (visible_count + 1e-6)[None, :, :]
        weighted_cost[visible_count[None].repeat(3, 1, 1) == 0] = 0
        
        return weighted_cost, visible_count > 0

    def _extract_semantic_feature_by_path(self, index, images, file_list=None):
        """
        Extract semantic features from the in-memory resized image tensor.
        This restores the original semantic feature generation behavior.
        """
        img_np = images[0][index].permute(1, 2, 0).cpu().numpy()
        feat = self.semantic_model.extract_features(img_np)
        return feat

    def _select_pose_closest_cross_video_views(self, index, poses, similarity_weights,
                                               img_1_length, img_2_length, num_closest_views=3):
        """
        Select top-k closest views from the opposite video based on camera-center
        translation distance (pose distance), after visibility filtering from
        non-zero similarity weights.
        """
        n_total = img_1_length + img_2_length
        if index < img_1_length:
            candidates = torch.arange(img_1_length, n_total, device=poses[0].device)
        else:
            candidates = torch.arange(0, img_1_length, device=poses[0].device)

        if candidates.numel() == 0:
            return []

        # Keep only views considered visible/similar enough by similarity weights.
        weights = similarity_weights[index, candidates]
        candidates = candidates[weights > 0]
        if candidates.numel() == 0:
            return []

        k = max(1, int(num_closest_views))
        k = min(k, candidates.numel())

        centers = poses[0][:, :3, 3]
        target_center = centers[index]
        candidate_centers = centers[candidates]
        dists = torch.norm(candidate_centers - target_center.unsqueeze(0), dim=1)

        topk_idx = torch.argsort(dists)[:k]
        selected = candidates[topk_idx]
        return [int(i.item()) for i in selected]
    
    def _voxelize_and_extract_features(self, point_map, total_cost_maps, total_visible_maps,
                                       images, img_1_length, img_2_length, voxel_size, H, W):
        """Voxelize point clouds and extract semantic features."""
        # Process point clouds
        pc1_results = self._process_point_cloud_costs(
            point_map[:img_1_length], total_cost_maps[:img_1_length],
            total_visible_maps[:img_1_length], voxel_size, H, W
        )
        pc2_results = self._process_point_cloud_costs(
            point_map[img_1_length:], total_cost_maps[img_1_length:],
            total_visible_maps[img_1_length:], voxel_size, H, W
        )
        
        # Extract DINOv3 features
        pc1_dino_feat = self.aggregate_features_to_voxels(
            images[0][:img_1_length], pc1_results['voxel_indices'], H, W
        )
        pc2_dino_feat = self.aggregate_features_to_voxels(
            images[0][img_1_length:], pc2_results['voxel_indices'], H, W
        )
        
        return pc1_results, pc2_results, pc1_dino_feat, pc2_dino_feat
    
    def _process_point_cloud_costs(self, point_map, cost_maps, visible_masks, voxel_size, H, W):
        """Process costs for a point cloud through voxelization."""
        points = point_map.reshape(-1, 3)
        points_tensor = torch.tensor(points, device=self.device)
        n_points = points_tensor.shape[0]

        # 3) Ignore points outside the normalized scene cube during voxelization.
        in_cube_mask = torch.ones(n_points, dtype=torch.bool, device=self.device)
        cube_radius_scaled = getattr(self.geometry_model, 'scene_cube_radius_scaled', None)
        if cube_radius_scaled is not None:
            in_cube_mask = (points_tensor.abs() <= (cube_radius_scaled + 1e-6)).all(dim=1)

        if in_cube_mask.sum() == 0:
            return {
                'cost_maps': torch.zeros(n_points, device=self.device),
                'voxel_indices': torch.zeros(n_points, dtype=torch.long, device=self.device),
                'voxel_locations': torch.zeros((1, 3), device=self.device)
            }
        
        # Voxelization
        point_to_voxel_in_cube = voxelization(
            points_tensor[in_cube_mask][None], voxel_size
        )[0]

        # Map full point set to voxel ids, default 0 for out-of-cube points.
        point_to_voxel = torch.zeros(n_points, dtype=torch.long, device=self.device)
        point_to_voxel[in_cube_mask] = point_to_voxel_in_cube
        
        # Aggregate costs to voxels
        visible_points = visible_masks.view(-1).to(self.device) & in_cube_mask
        visible_costs = cost_maps.view(-1)[visible_points]
        visible_voxel_indices = point_to_voxel[visible_points]

        if visible_voxel_indices.numel() > 0:
            cost_per_voxel = scatter_mean(visible_costs, visible_voxel_indices, dim=0)
            max_voxel_id = point_to_voxel_in_cube.max()
            full_cost_map = torch.zeros(max_voxel_id + 1, device=self.device)
            full_cost_map[:cost_per_voxel.shape[0]] = cost_per_voxel
            point_costs = torch.zeros(n_points, device=self.device)
            point_costs[in_cube_mask] = full_cost_map[point_to_voxel_in_cube]
        else:
            point_costs = torch.zeros(n_points, device=self.device)

        # Compute voxel locations only from in-cube points.
        voxel_locations = scatter_mean(
            points_tensor[in_cube_mask], point_to_voxel_in_cube, dim=0
        )
        
        return {
            'cost_maps': point_costs,
            'voxel_indices': point_to_voxel,
            'voxel_locations': voxel_locations
        }

    def _get_in_cube_point_mask(self, point_map):
        """Return per-point in-cube mask using normalized cube radius."""
        cube_radius_scaled = getattr(self.geometry_model, 'scene_cube_radius_scaled', None)
        if cube_radius_scaled is None:
            return None
        points = torch.tensor(point_map, device=self.device)
        return (points.abs() <= (cube_radius_scaled + 1e-6)).all(dim=-1)
    
    def _compute_cluster_costs(self, pc1_results, pc2_results, masks_list, total_visible_maps,
                               img_1_length, img_2_length, H, W):
        """Aggregate voxel costs to SAM regions."""
        cluster_cost_1 = self._compute_cluster_costs_single(
            pc1_results['cost_maps'], masks_list[:img_1_length],
            total_visible_maps, img_1_length, H, W, offset=0
        )
        cluster_cost_2 = self._compute_cluster_costs_single(
            pc2_results['cost_maps'], masks_list[img_1_length:],
            total_visible_maps, img_2_length, H, W, offset=img_1_length
        )
        
        return cluster_cost_1, cluster_cost_2
    
    def _compute_cluster_costs_single(self, cost_maps, masks_list, total_visible_maps,
                                      n_images, H, W, offset=0):
        """Aggregate costs to regions for a single set."""
        cluster_costs = []
        
        for idx in range(n_images):
            cluster_cost = torch.zeros(H, W).to(self.device)
            sam_mask = torch.tensor(masks_list[idx]).cuda()
            cost_map = cost_maps.view(-1, H, W)[idx].cuda()
            visible_map = total_visible_maps.view(-1, H, W)[idx + offset].cuda()
            
            for region_id in sam_mask.unique():
                if region_id == -1:
                    continue
                
                region_mask = sam_mask == region_id
                valid_mask = region_mask & visible_map
                
                if (valid_mask.sum() / region_mask.sum()) > 0.3:
                    cluster_cost[region_mask] = cost_map[valid_mask].mean()
            
            cluster_costs.append(cluster_cost)
        
        return torch.stack(cluster_costs)
    
    def _detect_and_merge_objects(self, cluster_cost_1, cluster_cost_2, total_visible_maps,
                                  masks_list, pc1_results, pc2_results, pc1_dino_feat,
                                  pc2_dino_feat, img_1_length, img_2_length, H, W, voxel_size):
        """Detect objects by thresholding and merge across frames."""
        detection_config = self.config['detection']
        merging_config = self.config['merging']
        
        combined_cost = torch.cat([cluster_cost_1, cluster_cost_2], dim=0)
        threshold = self._compute_changed_region_threshold(combined_cost, total_visible_maps)
        # Detect objects in video 1
        object_list_1, mask_index_1, threshold_1 = self._detect_objects(
            cluster_cost_1,
            total_visible_maps[:img_1_length],
            masks_list[:img_1_length],
            pc1_results['voxel_locations'],
            pc1_results['voxel_indices'],
            pc1_dino_feat,
            img_1_length, 
            threshold,
            H, W
        )
        
        # Detect objects in video 2
        object_list_2, mask_index_2, threshold_2 = self._detect_objects(
            cluster_cost_2,
            total_visible_maps[img_1_length:],
            masks_list[img_1_length:],
            pc2_results['voxel_locations'],
            pc2_results['voxel_indices'],
            pc2_dino_feat,
            img_2_length,
            threshold,
            H, W
        )
        
        # Merge objects across frames
        object_id_list_1, merged_object_list_1 = merge_objects(
            object_list_1,
            geometry_distance_threshold=merging_config['geometry_distance_threshold_of_voxel_size'] * voxel_size,
            visual_threshold_ratio=merging_config['visual_threshold_ratio'],
            geometry_threshold_ratio=merging_config['geometry_threshold_ratio'],
            general_threshold=merging_config['general_threshold']
        )
        
        object_id_list_2, merged_object_list_2 = merge_objects(
            object_list_2,
            geometry_distance_threshold=merging_config['geometry_distance_threshold_of_voxel_size'] * voxel_size,
            visual_threshold_ratio=merging_config['visual_threshold_ratio'],
            geometry_threshold_ratio=merging_config['geometry_threshold_ratio'],
            general_threshold=merging_config['general_threshold']
        )
        
        return {
            'object_list_1': object_list_1,
            'object_list_2': object_list_2,
            'mask_index_1': mask_index_1,
            'mask_index_2': mask_index_2,
            'threshold_1': threshold_1,
            'threshold_2': threshold_2,
            'object_id_list_1': object_id_list_1,
            'object_id_list_2': object_id_list_2,
            'merged_objects_1': merged_object_list_1,
            'merged_objects_2': merged_object_list_2
        }
    
    def _compute_changed_region_threshold(self, combined_cost, visible_maps):
        '''
        Compute the threshold for changed regions. 
        If change_region_threshold is not specified, compute the threshold using the visible cost values.
        '''
        detection_config = self.config['detection']
        
        # Get visible cost values.
        # For very large tensors, deterministically decimate spatially (2x2 stride)
        # before masking to reduce memory/compute without random resampling noise.
        if combined_cost.numel() > 8_000_000 and combined_cost.ndim >= 2:
            slicer = [slice(None)] * combined_cost.ndim
            slicer[-2] = slice(None, None, 2)
            slicer[-1] = slice(None, None, 2)
            ds_combined_cost = combined_cost[tuple(slicer)]
            ds_visible_maps = visible_maps[tuple(slicer)]
            visible_costs = ds_combined_cost[ds_visible_maps]
            if visible_costs.numel() == 0:
                visible_costs = combined_cost[visible_maps]
        else:
            visible_costs = combined_cost[visible_maps]
        if visible_costs.numel() == 0:
            if detection_config['change_region_threshold'] is not None:
                return detection_config['change_region_threshold']
            return float(combined_cost.mean().item())
        
        # Compute threshold
        if detection_config['change_region_threshold'] is not None:
            threshold = detection_config['change_region_threshold']
        else:
            quantile_val = self._safe_quantile(
                visible_costs, detection_config['filter_percentage_before_threshold']
            )
            filtered_costs = visible_costs[visible_costs > quantile_val]
            if filtered_costs.numel() == 0:
                return float(quantile_val)
            
            if detection_config['threshold_method'] == 'otsu':
                threshold = threshold_otsu(filtered_costs.view(-1)[None].cpu().numpy())
            elif detection_config['threshold_method'] == 'max_entropy':
                threshold = threshold_maximum_entropy(filtered_costs)
            else:
                raise ValueError(f"Invalid threshold method: {detection_config['threshold_method']}")
            
        return threshold

    def _safe_quantile(self, values, q, max_exact_elements=8_000_000, sample_elements=4_000_000):
        """
        Quantile computation that avoids RuntimeError on very large tensors.

        For very large inputs, estimate quantile from a deterministic subsample
        and compute the quantile on CPU for better robustness.
        """
        flat = values.reshape(-1)
        q = float(q)
        q = min(max(q, 0.0), 1.0)

        if flat.numel() <= max_exact_elements:
            try:
                return float(torch.quantile(flat, q).item())
            except RuntimeError:
                # Fall back to CPU if backend kernel fails.
                return float(torch.quantile(flat.float().cpu(), q).item())

        # For huge tensors, deterministically subsample to avoid oversized kernels.
        if flat.numel() > sample_elements:
            step = max(1, flat.numel() // sample_elements)
            flat = flat[::step]
            if flat.numel() > sample_elements:
                flat = flat[:sample_elements]

        return float(torch.quantile(flat.float().cpu(), q).item())
    
    def _detect_objects(self, cluster_cost_maps, visible_maps, masks_list,
                       voxel_locations, voxel_indices, dino_features,
                       img_length, threshold, H, W):
        """Detect objects by thresholding cost maps."""
        detection_config = self.config['detection']
        
        # Threshold to get object masks
        object_masks = cluster_cost_maps > threshold
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
                if region_mask.sum() < detection_config['min_detection_pixel']:
                    continue
                
                # Extract voxel points and features
                voxel_ids = voxel_indices.view(-1, H, W)[idx][region_mask]
                points = voxel_locations[voxel_ids].unique(dim=0).view(-1, 3)
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
    
    def _associate_and_save_objects(self, object_results, images, cluster_cost_1, cluster_cost_2,
                                    img_1_length, img_2_length, H, W, output_dir):
        """Associate objects between videos and save masks."""
        # Compute object similarity matrix
        merged_1 = object_results['merged_objects_1']
        merged_2 = object_results['merged_objects_2']
        
        object_sim_matrix = torch.zeros(len(merged_1), len(merged_2))
        for i, obj1 in enumerate(merged_1):
            for j, obj2 in enumerate(merged_2):
                object_sim_matrix[i, j] = self.semantic_model.compute_similarity(
                    obj1['dino_feat'], obj2['dino_feat']
                )
        
        # Create unified IDs
        unit_ids = self._create_unified_object_ids(
            object_results['object_id_list_1'],
            object_results['object_id_list_2'],
            object_sim_matrix
        )
        
        # Prepare colors
        colors = self._create_color_palette()
        max_score = self.config['detection']['max_score']
        
        # Save object masks
        object_masks = {'H': H, 'W': W}
        
        masks_1 = self._save_object_masks(
            object_results['object_id_list_1'],
            object_results['mask_index_1'],
            unit_ids, images, cluster_cost_1,
            img_1_length, H, W, colors, max_score, 'video_1', output_dir
        )
        
        masks_2 = self._save_object_masks(
            object_results['object_id_list_2'],
            object_results['mask_index_2'],
            unit_ids, images, cluster_cost_2,
            img_2_length, H, W, colors, max_score, 'video_2', output_dir, img_1_length
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
        
        # Save to file
        output_masks_file = output_dir / 'object_masks.pkl'
        with open(output_masks_file, 'wb') as f:
            pickle.dump(object_masks, f)
        
        return object_masks, unit_ids
    
    def _create_unified_object_ids(self, object_id_list_1, object_id_list_2, object_sim_matrix):
        """Create unified object IDs across two video sequences."""
        similarity_threshold = self.config['merging']['object_similarity_threshold']
        
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
            
            if max_sim > similarity_threshold:
                matched_obj_id = object_id_list_1.unique()[similarity_scores.argmax()]
                unit_ids['video_2'][obj_id.item()] = unit_ids['video_1'][matched_obj_id.item()]
            else:
                unit_ids['video_2'][obj_id.item()] = next_id
                next_id += 1
        
        return unit_ids
    
    def _save_object_masks(self, object_id_list, mask_indices, unit_ids, images,
                          total_cluster_cost_maps, current_sequence_length, H, W, colors,
                          max_score, video_key, output_dir, first_sequence_length=0):
        """Save object masks and visualizations."""
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
                
                for frame_idx in range(current_sequence_length):
                    frame_mask = single_mask.view(current_sequence_length, -1)[frame_idx].view(H, W)
                    
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
                    if self.config['visualization']['save_detections']:
                        img_offset = 0 if video_key == 'video_1' else first_sequence_length
                        rgb_image = images[0][frame_idx + img_offset].permute(1, 2, 0).cpu().numpy()
                        rgb_image[frame_mask.cpu().numpy().astype(bool)] = color.cpu().numpy()
                        vis_path = vis_dir / f'{video_key}_frame_{frame_idx}_obj_{object_id}.png'
                        plt.imsave(vis_path, rgb_image)
        
        return object_masks

    def _project_points_to_image_mask(self, points_world, pose, intrinsic, H, W):
        """Project 3D world points to one image and return a binary mask."""
        if points_world is None or points_world.numel() == 0:
            return torch.zeros((H, W), dtype=torch.bool, device=self.device)

        points_world = points_world.to(self.device).reshape(-1, 3)
        points_h = torch.cat(
            [points_world, torch.ones_like(points_world[..., :1])], dim=-1
        )
        points_cam = (points_h @ pose.cuda().inverse().T)[..., :3]
        points_img_h = points_cam @ intrinsic.cuda().T

        z = points_img_h[:, 2]
        points_img = points_img_h[:, :2] / (z[:, None] + 1e-8)

        valid = (
            (z > 0)
            & (points_img[:, 0] >= 0) & (points_img[:, 0] < W)
            & (points_img[:, 1] >= 0) & (points_img[:, 1] < H)
        )

        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        if valid.any():
            valid_points = points_img[valid]
            xs = valid_points[:, 0].long().clamp(0, W - 1)
            ys = valid_points[:, 1].long().clamp(0, H - 1)
            mask[ys, xs] = True
        return mask

    def _save_video2_difference_masks(self, object_masks, unit_ids,
                                      poses, intrinsics, img_1_length, img_2_length,
                                      point_map, H, W, output_dir):
        """
        Save per-frame binary difference masks in video_2.

        The mask contains:
        1) all detected changed regions in video_2;
        2) projected points of removed objects (objects only detected in video_1).
        """
        diff_dir = output_dir / 'video_2_difference_masks'
        diff_dir.mkdir(parents=True, exist_ok=True)

        video1_uids = set(unit_ids['video_1'].values())
        video2_uids = set(unit_ids['video_2'].values())
        all_uids = video1_uids

        # Recover dense removed-object point clouds from video_1 masks.
        # This avoids sparse projection caused by voxel-center-only point sets.
        removed_dense_points = {}
        for uid in all_uids:
            obj_data = object_masks.get(uid, {})
            video1_frames = obj_data.get('video_1', {})
            dense_points_list = []

            for frame_idx, frame_data in video1_frames.items():
                if frame_idx >= img_1_length:
                    continue
                frame_rle = frame_data['mask']
                frame_mask = mask_utils.decode(frame_rle).astype(bool)
                if not frame_mask.any():
                    continue

                frame_points = point_map[frame_idx][frame_mask]
                if frame_points.size == 0:
                    continue
                dense_points_list.append(torch.from_numpy(frame_points).float())

            if len(dense_points_list) > 0:
                removed_dense_points[uid] = torch.cat(dense_points_list, dim=0).to(self.device)

        # Cache projected removed masks per frame for efficiency.
        projected_removed_masks = []
        for frame_idx in range(img_2_length):
            pose_2 = poses[img_1_length + frame_idx]
            intrinsic_2 = intrinsics[img_1_length + frame_idx]
            removed_mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)

            for uid in all_uids:
                if uid not in removed_dense_points:
                    continue
                removed_points = removed_dense_points[uid]
                removed_mask |= self._project_points_to_image_mask(
                    removed_points, pose_2, intrinsic_2, H, W
                )
            projected_removed_masks.append(removed_mask)

        for frame_idx in range(img_2_length):
            frame_mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)

            for _, obj_data in object_masks.items():
                if not isinstance(obj_data, dict):
                    continue
                video2_frames = obj_data.get('video_2', None)
                if not isinstance(video2_frames, dict):
                    continue

                # Be tolerant to frame-key types (int/np.int/str).
                matched_frame_data = None
                for frame_key, frame_data in video2_frames.items():
                    try:
                        if int(frame_key) == frame_idx:
                            matched_frame_data = frame_data
                            break
                    except (TypeError, ValueError):
                        continue
                if matched_frame_data is None:
                    continue

                frame_rle = matched_frame_data['mask']
                frame_mask |= torch.tensor(
                    mask_utils.decode(frame_rle).astype(bool), device=self.device
                )

            frame_mask |= projected_removed_masks[frame_idx]
            out_path = diff_dir / f'frame_{frame_idx:05d}.png'
            cv2.imwrite(str(out_path), frame_mask.cpu().numpy().astype(np.uint8) * 255)
    
    def _create_color_palette(self):
        """Create a color palette for object visualization."""
        cmap1 = plt.cm.tab20
        cmap2 = plt.cm.tab20b
        cmap3 = plt.cm.tab20c
        
        colors1 = torch.tensor([cmap1(i)[:3] for i in range(20)]).float().to(self.device)
        colors2 = torch.tensor([cmap2(i)[:3] for i in range(20)]).float().to(self.device)
        colors3 = torch.tensor([cmap3(i)[:3] for i in range(20)]).float().to(self.device)
        
        return torch.cat([colors1, colors2, colors3], dim=0)
    
    def _save_visualizations(self, point_map, pc1_results, pc2_results, images,
                            object_results, unit_ids, img_1_length, output_dir):
        """Save point cloud visualizations."""
        # Cost map visualizations
        total_costs = torch.cat([pc1_results['cost_maps'], pc2_results['cost_maps']])
        cost_vis = apply_colormap(total_costs).reshape(len(point_map), -1, 3)
        in_cube_mask = self._get_in_cube_point_mask(point_map)
        if in_cube_mask is not None:
            in_cube_mask_np = in_cube_mask.detach().cpu().numpy().reshape(len(point_map), -1)
        else:
            in_cube_mask_np = np.ones((len(point_map), point_map.shape[1] * point_map.shape[2]), dtype=bool)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_map.reshape(-1, 3)[in_cube_mask_np.reshape(-1)])
        pcd.colors = o3d.utility.Vector3dVector(cost_vis.reshape(-1, 3)[in_cube_mask_np.reshape(-1)])
        o3d.io.write_point_cloud(str(output_dir / 'cost_map_merged.ply'), pcd)

        pcd1 = o3d.geometry.PointCloud()
        pcd1_mask = in_cube_mask_np[:img_1_length].reshape(-1)
        pcd1.points = o3d.utility.Vector3dVector(point_map[:img_1_length].reshape(-1, 3)[pcd1_mask])
        pcd1.colors = o3d.utility.Vector3dVector(cost_vis[:img_1_length].reshape(-1, 3)[pcd1_mask])
        o3d.io.write_point_cloud(str(output_dir / 'cost_map_1.ply'), pcd1)

        pcd2 = o3d.geometry.PointCloud()
        pcd2_mask = in_cube_mask_np[img_1_length:].reshape(-1)
        pcd2.points = o3d.utility.Vector3dVector(point_map[img_1_length:].reshape(-1, 3)[pcd2_mask])
        pcd2.colors = o3d.utility.Vector3dVector(cost_vis[img_1_length:].reshape(-1, 3)[pcd2_mask])
        o3d.io.write_point_cloud(str(output_dir / 'cost_map_2.ply'), pcd2)
        
        # Object instance visualizations
        colors = self._create_color_palette()
        self._save_object_point_clouds(
            point_map[:img_1_length].reshape(-1, 3),
            point_map[img_1_length:].reshape(-1, 3),
            object_results['object_id_list_1'],
            object_results['object_id_list_2'],
            object_results['mask_index_1'],
            object_results['mask_index_2'],
            images[0][:img_1_length],
            images[0][img_1_length:],
            unit_ids,
            colors, output_dir
        )
    
    def _save_object_point_clouds(self, pc1_points, pc2_points, object_id_list_1, object_id_list_2,
                                  mask_index_1, mask_index_2, images_1, images_2, unit_ids, colors, output_dir):
        """Save point clouds colored by object ID."""
        vis_pc_1 = images_1.permute(0, 2, 3, 1).reshape(-1, 3)
        vis_pc_2 = images_2.permute(0, 2, 3, 1).reshape(-1, 3)
        
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
    
    def _save_ground_truth_visualization(self, scene_name, images, img_1_length, H, W, output_dir):
        """Save ground truth object visualizations."""
        gt_dir = Path(self.config['dataset']['gt_dir'])
        gt_file_path = gt_dir / scene_name / 'segments.pkl'
        resample_rate = self.config['dataset']['resample_rate']
        
        if not os.path.exists(gt_file_path):
            return
        
        gt = pickle.load(open(gt_file_path, 'rb'))
        colors = self._create_color_palette()
        
        # Similar implementation as the original save_ground_truth_visualization
        # ...abbreviated for brevity
    
    def _get_cache_search_root(self, output_dir):
        """Use nearest max_size_* ancestor as cache root when present."""
        out_dir = Path(output_dir)
        for candidate in [out_dir] + list(out_dir.parents):
            if candidate.name.startswith('max_size_'):
                return candidate
        return out_dir.parent

    def _load_cached_cost_maps(self, output_dir, scene_name, scene_group=None):
        """Load cached cost maps if available."""
        output_dir = Path(output_dir)
        root_dir = self._get_cache_search_root(output_dir)

        candidate_scene_dirs = []
        seen = set()

        def add_candidate(path):
            p = Path(path)
            if not p.is_dir():
                return
            key = str(p.resolve())
            if key in seen:
                return
            candidate_scene_dirs.append(p)
            seen.add(key)

        # Prefer current directory first.
        add_candidate(output_dir)

        # Then scan sibling parameter folders under the same max_size root.
        if scene_group is not None:
            for scene_dir in root_dir.glob(f'**/{scene_group}/{scene_name}'):
                add_candidate(scene_dir)
        else:
            for scene_dir in root_dir.glob(f'**/{scene_name}'):
                add_candidate(scene_dir)

        for scene_dir in candidate_scene_dirs:
            cost_path = scene_dir / 'cost_maps.pkl'
            visible_path = scene_dir / 'visible_maps.pkl'
            if cost_path.exists() and visible_path.exists():
                with open(cost_path, 'rb') as f:
                    cost_maps = pickle.load(f)
                with open(visible_path, 'rb') as f:
                    visible_maps = pickle.load(f)
                print(f"Loaded cached cost maps from: {scene_dir}")
                return (cost_maps, visible_maps)

        return None
    
    def _save_cost_maps(self, output_dir, cost_maps, visible_maps):
        """Save cost maps to disk."""
        with open(output_dir / 'cost_maps.pkl', 'wb') as f:
            pickle.dump(cost_maps, f)
        with open(output_dir / 'visible_maps.pkl', 'wb') as f:
            pickle.dump(visible_maps, f)
    
    def cleanup(self):
        """Clean up all model resources."""
        self.geometry_model.cleanup()
        self.mask_model.cleanup()
        self.semantic_model.cleanup()
        torch.cuda.empty_cache()
