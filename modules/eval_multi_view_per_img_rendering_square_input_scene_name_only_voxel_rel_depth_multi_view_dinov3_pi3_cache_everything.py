import sys
sys.path.append('/home/yuqun/research/scene_change/Pi3')
sys.path.append('/u/ywu20/scene_change/segment_ls')
from pi3.utils.basic import load_images_as_tensor_from_list, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import torchvision.ops as ops
import subprocess
from copy import deepcopy
import copy
from torch_scatter import scatter_mean
from skimage.filters import threshold_otsu
import faiss

import sys
sys.path.append('/home/yuqun/research/feed_forward/region_slot/')
from pathlib import Path 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
# from utils import open_file
import torch
from pycocotools import mask as mask_utils
import scipy
from sklearn.metrics import average_precision_score, precision_recall_curve
# notebook tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import torch.nn.functional as F
import cv2
from pathlib import Path
# visualize a few matches
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl


sys.path.append('/home/yuqun/research/scene_change/MoGe/moge/utils')
from geometry_torch import recover_focal_shift

# Estiamte the masks for the image
from segment_anything_ls import build_sam, SamAutomaticMaskGenerator
import tempfile
import torchvision.transforms as TF
import torch.nn.functional as NF
import math 
import itertools
import pickle
import os
from voc_eval import BoxList, eval_detection_voc
import matplotlib.pyplot as plt
from matplotlib import cm
from utils_two_view import *
import time
import open3d as o3d

device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.bfloat16  # or torch.float16


# Need to define a different confidence for the pi3 model
def per_single_img_confidence(single_image_1, model, threshold=0.4, dtype=torch.bfloat16, return_depth=False):
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(single_image_1[None]) # Add batch dimension
    
    depth_conf = res['conf'][..., 0] #
    depth_map = res['local_points'][..., 2]
    point_map_by_unprojection = res['points'] # (N, H, W, 3)

    percentage = threshold

    conf_max_1 = torch.kthvalue(depth_conf.reshape(-1), int(percentage * depth_conf.reshape(-1).numel())).values.item()

    # conf_max_1 = 1.5
    # conf_max_2 = 1.5

    combined_max = conf_max_1
    H, W = depth_conf.shape[-2:]

    conf_1 = (depth_conf / combined_max).clamp(0, 1).view(H, W)

    if return_depth:
        return conf_1, depth_map
    
    return conf_1

def threshold_maximum_entropy(image, nbins=256):
    """
    Compute threshold using Maximum Entropy method (Kapur's method) - PyTorch CUDA implementation.
    
    Args:
        image: torch.Tensor or numpy array, input image or flattened array
        nbins: number of histogram bins
    
    Returns:
        threshold value (float)
    """
    # Convert to torch tensor if numpy and move to same device
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
    
    # Flatten if necessary
    image = image.flatten()
    
    # Compute min/max for histogram range
    min_val = image.min()
    max_val = image.max()
    
    # Compute histogram on GPU
    hist = torch.histc(image.float(), bins=nbins, min=min_val.item(), max=max_val.item())
    
    # Normalize histogram to get probabilities
    hist = hist.float()
    hist = hist / hist.sum()
    
    # Avoid log(0) by adding small epsilon
    epsilon = torch.finfo(torch.float32).eps
    
    max_entropy = torch.tensor(-float('inf'), device=image.device)
    best_threshold = torch.tensor(0.0, device=image.device)
    
    # Compute bin width for threshold values
    bin_width = (max_val - min_val) / nbins
    
    # Try all possible thresholds on GPU
    for t in range(1, nbins):
        # Background (class 0)
        P0 = hist[:t].sum()
        if P0 == 0:
            continue
        
        # Foreground (class 1)
        P1 = hist[t:].sum()
        if P1 == 0:
            continue
        
        # Entropy of background
        hist_bg = hist[:t] / (P0 + epsilon)
        H0 = -(hist_bg * torch.log(hist_bg + epsilon)).sum()
        
        # Entropy of foreground
        hist_fg = hist[t:] / (P1 + epsilon)
        H1 = -(hist_fg * torch.log(hist_fg + epsilon)).sum()
        
        # Total entropy
        total_entropy = H0 + H1
        
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            best_threshold = min_val + t * bin_width
    
    return best_threshold.item()

def predict_bbox(file_list, img_1_length, img_2_length, output_dir, norm_to_ori_range=True, visible_percentage=0.5, args=None, model=None):
    # import pdb; pdb.set_trace()
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"Starting processing for scene: {Path(file_list[0]).parents[1].stem}")
    print(f"{'='*80}\n")
    
    stage_start = time.time()
    print(f"[Stage 1] Image loading and preprocessing: {time.time() - stage_start:.2f}s")

    appearance_feature_dim = 1280

    stage_start = time.time()
    
    images = load_images_as_tensor_from_list(file_list)[None].to(device) # (1, N, 3, H, W)
    print(f"[Stage 1] Image loading and preprocessing: {time.time() - stage_start:.2f}s")
    H, W = images.shape[-2:]
    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(images) # Add batch dimension
    
    depth_map = res['local_points'][..., 2][..., None] # B, N, H, W, 1
    depth_conf = res['conf'][..., 0]
    point_map_by_unprojection = res['points'] # (N, H, W, 3)


    intrinsic = []
    for i in range(len(file_list)):
        local_point_map = res['local_points'][0, i]
        masks = torch.sigmoid( res['conf'][0, i]) > 0.1
        original_height, original_width = local_point_map.shape[-3:-1]
        aspect_ratio = original_width / original_height
        # use recover_focal_shift function from MoGe
        focal, shift = recover_focal_shift(local_point_map, masks)
        # Convert focal from normalized (relative to half-diagonal) to pixel units
        fx = focal * W / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal * H / 2 * (1 + aspect_ratio ** 2) ** 0.5
        intrinsic.append(torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]]).cuda())
    intrinsic = torch.stack(intrinsic, dim=0)[None] # 1, N, 3, 3
    print(f"[Stage 2] Model inference (cameras, depth, points): {time.time() - stage_start:.2f}s")
    poses = res['camera_poses']

    pc_voxel_size = get_robust_voxel_size(point_map_by_unprojection.reshape(-1, 3), subsample_size=1000000, scale_factor=300)
    scale = 1 / (pc_voxel_size / 0.005)

    depth_map = depth_map * scale
    point_map_by_unprojection = point_map_by_unprojection[0].cpu().numpy() * scale
    poses[:, :, :3, 3] = poses[:, :, :3, 3] * scale

    grid_intrinsic = torch.tensor([
        2.0 / W, 0, -1,
        0, 2.0 / H, -1
    ]).reshape(2, 3).to(device)

    img_coors = get_img_coor(H, W).to(device)

    stage_start = time.time()
    similarity_matrix = -1 * torch.ones(len(file_list), len(file_list)).cuda() # -1 means no similarity
    set_2_mapper = lambda x: x + img_1_length
    # From set 1 to set 2
    for i in range(img_1_length):
        for j in range(img_2_length):
            set_2_idx = set_2_mapper(j) 
            similarity_1 = calculate_mask_percentage(poses[0][i], poses[0][set_2_idx], depth_map[0][i].permute(2,0,1), depth_map[0][set_2_idx].permute(2,0,1), intrinsic[0][i], intrinsic[0][set_2_idx], grid_intrinsic, 0.3, img_coors, H, W) 
            similarity_2 = calculate_mask_percentage(poses[0][set_2_idx], poses[0][i], depth_map[0][set_2_idx].permute(2,0,1), depth_map[0][i].permute(2,0,1), intrinsic[0][set_2_idx], intrinsic[0][i], grid_intrinsic, 0.3, img_coors, H, W)
            similarity_matrix[i, set_2_idx] = (similarity_1 + similarity_2) / 2
            similarity_matrix[set_2_idx, i] = similarity_matrix[i, set_2_idx]
    print(f"[Stage 3] Similarity matrix calculation: {time.time() - stage_start:.2f}s")
     

    # closest_indices = similarity_matrix.argmax(dim=1)
    percentage_threshold =visible_percentage
    # closest_indices = torch.argsort(similarity_matrix, dim=1, descending=True)[:, :top_k] # N, K
    similarity_weight_matrix = torch.zeros_like(similarity_matrix)
    for i in range(len(file_list)):
        for j in range(len(file_list)): # Already make sure that images in the same scene have -1similarity
            if similarity_matrix[i, j] > percentage_threshold:
                similarity_weight_matrix[i, j] = similarity_matrix[i, j]

            if similarity_weight_matrix[i].sum() == 0:
                j = torch.argmax(similarity_matrix[i])
                similarity_weight_matrix[i, j] = 1
    
    similarity_weight_matrix = similarity_weight_matrix / similarity_weight_matrix.sum(dim=1, keepdim=True) # N, N
    similarity_weight_matrix = similarity_weight_matrix.to(device)

    visible_matrix = (similarity_weight_matrix > 0).sum(dim=1)

    stage_start = time.time()
    # First store the confidence of each image
    conf_list = []
    for i in range(len(file_list)):
        conf_1 = per_single_img_confidence(copy.deepcopy(images[0][i:i+1]), model, threshold=0.1)
        conf_list.append([conf_1])
    # conf_list = torch.tensor(conf_list).cuda() # N, 2
    del model
    torch.cuda.empty_cache() # Save gpu memory
    print(f"[Stage 4] Confidence calculation: {time.time() - stage_start:.2f}s")

    # import pdb; pdb.set_trace()

    output_name = Path(file_list[0]).parents[1].stem # derek_working_room_1
    file_output_dir = Path(output_dir) / output_name
    Path(file_output_dir).mkdir(parents=True, exist_ok=True)    
    stage_start = time.time()
    input_dir = Path(file_list[0]).parents[1]
    mask_file_path = input_dir / "sam_masks.pkl"
    if os.path.exists(mask_file_path):
    # if False:
        with open(mask_file_path, 'rb') as f:
            masks_list = pickle.load(f)
        print(f"[Stage 5] SAM masks loaded from cache: {time.time() - stage_start:.2f}s")
    else:
        sam_checkpoint = "/home/yuqun/research/multi_purpose_nerf/dino_sam_scannet_save/segment_anything_main/checkpoints/sam_vit_h_4b8939.pth"
        mask_generator = SamAutomaticMaskGenerator(
            model=build_sam(checkpoint=sam_checkpoint).to(device="cuda"),
            points_per_side=16,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )        
        masks_list = []
        for i in range(len(file_list)):
            image_i = np.array(Image.open(file_list[i])).astype(np.uint8)
            mask_i = predict_mask(image_i, mask_generator)
            # Resize
            # mask_i = cv2.resize(mask_i, (ori_W_end - ori_W_start, ori_H_end - ori_H_start))
            mask_i = cv2.resize(mask_i.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)

            # Make sure that the resizing do not kill any index, if so, reorder the mask
            mask_i = reorder_mask(mask_i)
            
            masks_list.append(mask_i)
        with open(mask_file_path, 'wb') as f:
            pickle.dump(masks_list, f)
        del mask_generator
        torch.cuda.empty_cache()
        print(f"[Stage 5] SAM masks generation: {time.time() - stage_start:.2f}s")

    rand_colors = np.random.rand(100, 3)
    for mask_index, single_mask in enumerate(masks_list):
        single_mask = cv2.resize(single_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        single_mask_vis = np.zeros((H, W, 3))
        for i in np.unique(single_mask):
            if i == -1:
                continue
            mask_i = single_mask == i
            single_mask_vis[mask_i] = rand_colors[i]
        plt.imsave(f'{file_output_dir}/single_mask_vis_{mask_index}.png', single_mask_vis)


    stage_start = time.time()
    dinov3_vith16plus = torch.hub.load('/home/yuqun/research/feed_forward/dinov3','dinov3_vith16plus', source='local', weights="/home/yuqun/.cache/torch/hub/checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", ).to(device)
    dinov3_vith16plus.eval()
    print(f"[Stage 6] DINOv3 model loading: {time.time() - stage_start:.2f}s")
    total_cost_map_list = []
    total_visible_map_list = []


    lambda_dinov1_region_match = args.lambda_dino_region_match
    lambda_dinov1_rendering = args.lambda_dino
    lambda_foreground_geo = args.lambda_geo

    # Define the base filename patterns (without directory)
    if args.use_view_coverage_map and args.use_non_occluded_map:
        cost_map_filename = f'pi3_masks_fix_foreground_mask_total_cost_map_list_view_coverage_and_non_occluded.pkl'
        visible_map_filename = f'pi3_masks_fix_foreground_mask_total_visible_map_list_view_coverage_and_non_occluded.pkl'
    elif args.use_view_coverage_map:
        cost_map_filename = f'pi3_masks_fix_foreground_mask_total_cost_map_list_view_coverage.pkl'
        visible_map_filename = f'pi3_masks_fix_foreground_mask_total_visible_map_list_view_coverage.pkl'
    elif args.use_non_occluded_map:
        cost_map_filename = f'pi3_masks_fix_foreground_mask_total_cost_map_list_non_occluded.pkl'
        visible_map_filename = f'pi3_masks_fix_foreground_mask_total_visible_map_list_non_occluded.pkl'
    else:
        raise ValueError(f"Invalid use_view_coverage_map and use_non_occluded_map: {args.use_view_coverage_map} and {args.use_non_occluded_map}")

    # Search for cached files in root_dir (parent of output_dir) across all experiments
    root_dir = Path(output_dir).parent
    cached_cost_map_path = None
    cached_visible_map_path = None
    
    if args.use_cache:
        # Search in root_dir for any experiment directory containing the scene
        for exp_dir in root_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            scene_dir = exp_dir / output_name
            if not scene_dir.exists():
                continue
            
            cost_map_path = scene_dir / cost_map_filename
            visible_map_path = scene_dir / visible_map_filename
            
            if cost_map_path.exists() and visible_map_path.exists():
                cached_cost_map_path = cost_map_path
                cached_visible_map_path = visible_map_path
                print(f"Found cached cost maps in: {exp_dir.name}")
                break
    

    stage_start = time.time()
    if cached_cost_map_path and cached_visible_map_path and args.use_cache:
        with open(cached_cost_map_path, 'rb') as f:
            total_cost_map_list = pickle.load(f)
        with open(cached_visible_map_path, 'rb') as f:
            total_visible_map_list = pickle.load(f)
        print(f"[Stage 7] Cost maps loaded from cache: {time.time() - stage_start:.2f}s")
    else:
        print(f"[Stage 7] Computing cost maps for {len(file_list)} images...")
        for index in range(len(file_list)):
            img_start_time = time.time()
            total_cost_map_1_list = []
            total_visible_map_1_list = []
            total_non_occluded_map_1_list = []
            total_view_coverage_map_1_list = []

            img1_np = images[0][index].permute(1,2,0).cpu().numpy()
            sam_mask_1 = masks_list[index]
            sam_mask_1_merged = torch.tensor(sam_mask_1).cuda()
            pose_1 = poses[0][index].clone()
            depth_1 = depth_map[0][index].view(1, H, W).clone()
            intrinsic_1 = intrinsic[0][index].clone()
            conf_1 = conf_list[index][0]
            for closest_index in similarity_weight_matrix[index].argsort(dim=0, descending=True):
                if similarity_weight_matrix[index, closest_index] == 0:
                    break
                    
                # 1. Move complete operations to CPU where possible and free GPU tensors immediately
                img2_np = images[0][closest_index].permute(1,2,0).cpu().numpy()
                sam_mask_2 = masks_list[closest_index]        
                sam_mask_2_merged = torch.tensor(sam_mask_2).cuda()

                # 2. Create local variables for tensors instead of accessing the large tensors
                pose_2 = poses[0][closest_index].clone()
                depth_2 = depth_map[0][closest_index].view(1, H, W).clone()
                intrinsic_2 = intrinsic[0][closest_index].clone()
                
                # 3. Free intermediate results after using them
                vis_1 = valid_mask_after_proj(intrinsic_1.cuda(), pose_1.cuda(), depth_1, intrinsic_2.cuda(), pose_2.cuda(), depth_2, img_coors, H, W)
                vis_2 = valid_mask_after_proj(intrinsic_2.cuda(), pose_2.cuda(), depth_2, intrinsic_1.cuda(), pose_1.cuda(), depth_1, img_coors, H, W)

                foreground_dis_1 = foreground_distance(intrinsic_1.cuda(), pose_1.cuda(), depth_1, intrinsic_2.cuda(), pose_2.cuda(), depth_2, img_coors, grid_intrinsic, H, W)
                foreground_dis_2 = foreground_distance(intrinsic_2.cuda(), pose_2.cuda(), depth_2, intrinsic_1.cuda(), pose_1.cuda(), depth_1, img_coors, grid_intrinsic, H, W)

                # 4. Use local copies of confidence values (already done with conf_list)
                conf_2 = conf_list[closest_index][0]
                
                # 5. Process in smaller chunks
                foreground_1_pixel_geo = foreground_dis_1
                foreground_2_pixel_geo = foreground_dis_2

                foreground_1_pixel_geo_conf = compute_conf_norm_distance(foreground_1_pixel_geo, conf_1, norm_to_ori_range)
                foreground_2_pixel_geo_conf = compute_conf_norm_distance(foreground_2_pixel_geo, conf_2, norm_to_ori_range)

                # 6. Free GPU memory for tensors after they're no longer needed
                foreground_1_region_geo = torch.zeros_like(sam_mask_1_merged).float()
                foreground_2_region_geo = torch.zeros_like(sam_mask_2_merged).float()

                for i in sam_mask_1_merged.unique():
                    mask_i = sam_mask_1_merged == i
                    foreground_1_region_geo[mask_i] = (foreground_1_pixel_geo_conf * vis_1)[mask_i].mean()

                for i in sam_mask_2_merged.unique():
                    mask_i = sam_mask_2_merged == i
                    foreground_2_region_geo[mask_i] = (foreground_2_pixel_geo_conf * vis_2)[mask_i].mean()

                # Dinov1 feature
                dinov1_feat_1 = predict_dinov3_feat(img1_np, dinov3_vith16plus)
                dinov1_feat_2 = predict_dinov3_feat(img2_np, dinov3_vith16plus)

                dinov1_region_match_cost_1, dinov1_region_match_cost_2, _, _, _, _ = get_dino_matched_region_cost(dinov1_feat_1, dinov1_feat_2, sam_mask_1_merged, sam_mask_2_merged,
                                                                        vis_1, vis_2, foreground_1_pixel_geo, foreground_2_pixel_geo)

                # DINOv1 backward wrapping rendering
                dinov1_feat_1_reproject = reprojected_feature(intrinsic_1.cuda(), pose_1.cuda(), depth_1.view(1, H, W), intrinsic_2.cuda(), pose_2.cuda(), dinov1_feat_2, img_coors, grid_intrinsic, H, W)
                dinov1_feat_2_reproject = reprojected_feature(intrinsic_2.cuda(), pose_2.cuda(), depth_2.view(1, H, W), intrinsic_1.cuda(), pose_1.cuda(), dinov1_feat_1, img_coors, grid_intrinsic, H, W)

                dinov1_diff_1_occlusion = 1 - torch.cosine_similarity(dinov1_feat_1_reproject, dinov1_feat_1, dim=0)
                dinov1_diff_2_occlusion = 1 - torch.cosine_similarity(dinov1_feat_2_reproject, dinov1_feat_2, dim=0)

                occlusion_mask_1 = (foreground_1_pixel_geo < args.occlusion_threshold)
                occlusion_mask_2 = (foreground_2_pixel_geo < args.occlusion_threshold)

                dinov1_diff_1_occlusion[occlusion_mask_1] = 0
                dinov1_diff_2_occlusion[occlusion_mask_2] = 0

                dinov1_diff_region_1_occlusion = torch.zeros_like(sam_mask_1_merged).float()
                dinov1_diff_region_2_occlusion = torch.zeros_like(sam_mask_2_merged).float()

                for i in sam_mask_1_merged.unique():
                    if i == -1:
                        continue
                    mask_i = sam_mask_1_merged == i
                    if args.mean_over_non_occluded_region:
                        dinov1_diff_region_1_occlusion[mask_i] = dinov1_diff_1_occlusion[mask_i & ~occlusion_mask_1].mean() if (mask_i & ~occlusion_mask_1).sum() > 0 else 0
                    else:
                        dinov1_diff_region_1_occlusion[mask_i] = dinov1_diff_1_occlusion[mask_i & vis_1].mean() if (mask_i & vis_1).sum() > 0 else 0
                for i in sam_mask_2_merged.unique():
                    if i == -1:
                        continue
                    mask_i = sam_mask_2_merged == i
                    if args.mean_over_non_occluded_region:
                        dinov1_diff_region_2_occlusion[mask_i] = dinov1_diff_2_occlusion[mask_i & ~occlusion_mask_2].mean() if (mask_i & ~occlusion_mask_2).sum() > 0 else 0
                    else:
                        dinov1_diff_region_2_occlusion[mask_i] = dinov1_diff_2_occlusion[mask_i & vis_2].mean() if (mask_i & vis_2).sum() > 0 else 0

                foreground_geo_cost_1 = (foreground_1_region_geo * vis_1).cpu()
                foreground_geo_cost_2 = (foreground_2_region_geo * vis_2).cpu()

                dinov1_region_match_cost_1 = (dinov1_region_match_cost_1 * vis_1).cpu()
                dinov1_region_match_cost_2 = (dinov1_region_match_cost_2 * vis_2).cpu()

                dinov1_region_rendering_cost_1 = (dinov1_diff_region_1_occlusion * vis_1).cpu()
                dinov1_region_rendering_cost_2 = (dinov1_diff_region_2_occlusion * vis_2).cpu()

                total_cost_map_1_list.append(torch.stack([foreground_geo_cost_1, dinov1_region_match_cost_1, dinov1_region_rendering_cost_1], dim=0))
                total_visible_map_1_list.append((~occlusion_mask_1 & vis_1).cpu())
                total_non_occluded_map_1_list.append(~occlusion_mask_1.cpu())
                total_view_coverage_map_1_list.append(vis_1.cpu())


            # Free all GPU memory
            del sam_mask_1_merged, sam_mask_2_merged, foreground_1_region_geo, foreground_2_region_geo, dinov1_diff_region_1_occlusion, dinov1_diff_region_2_occlusion,
            del dinov1_region_match_cost_1, dinov1_region_match_cost_2, dinov1_region_rendering_cost_1, dinov1_region_rendering_cost_2, foreground_geo_cost_1, foreground_geo_cost_2
            del dinov1_feat_1, dinov1_feat_2, dinov1_feat_1_reproject, dinov1_feat_2_reproject, dinov1_diff_1_occlusion, dinov1_diff_2_occlusion
            torch.cuda.empty_cache()


            total_visible_map_1_list = torch.stack(total_visible_map_1_list) # N, H, W
            total_cost_map_1_list = torch.stack(total_cost_map_1_list) # N, 3, H, W
            total_non_occluded_map_1_list = torch.stack(total_non_occluded_map_1_list) # N, H, W
            total_view_coverage_map_1_list = torch.stack(total_view_coverage_map_1_list) # N, H, W
            # import pdb; pdb.set_trace()

            if args.use_view_coverage_map and args.use_non_occluded_map:
                total_visible_count_map_1 = (total_visible_map_1_list).sum(dim=0) # H, W
                weighted_total_cost_map_1 = (total_cost_map_1_list * (total_visible_map_1_list[:, None, :, :])).sum(dim=0) # 3, H, W
                weighted_total_cost_map_1 = weighted_total_cost_map_1 / (total_visible_count_map_1 + 1e-6)[None, :, :] # 3, H, W
                weighted_total_cost_map_1[total_visible_count_map_1[None].repeat(3, 1, 1) == 0] = 0 # 3, H, W
            elif args.use_view_coverage_map:
                total_visible_count_map_1 = (total_view_coverage_map_1_list).sum(dim=0) # H, W
                weighted_total_cost_map_1 = (total_cost_map_1_list * (total_view_coverage_map_1_list[:, None, :, :])).sum(dim=0) # 3, H, W
                weighted_total_cost_map_1 = weighted_total_cost_map_1 / (total_visible_count_map_1 + 1e-6)[None, :, :] # 3, H, W
                weighted_total_cost_map_1[total_visible_count_map_1[None].repeat(3, 1, 1) == 0] = 0 # 3, H, W
            elif args.use_non_occluded_map:
                total_visible_count_map_1 = (total_non_occluded_map_1_list).sum(dim=0)
                weighted_total_cost_map_1 = (total_cost_map_1_list * (total_non_occluded_map_1_list[:, None, :, :])).sum(dim=0) # 3, H, W
                weighted_total_cost_map_1 = weighted_total_cost_map_1 / (total_visible_count_map_1 + 1e-6)[None, :, :] # 3, H, W
                weighted_total_cost_map_1[total_visible_count_map_1[None].repeat(3, 1, 1) == 0] = 0 # 3, H, W
            total_cost_map_list.append(weighted_total_cost_map_1) # 3, H, W
            total_visible_map_list.append(total_visible_count_map_1 > 0) # H, W
            # Save the total cost map list and total occlusion map list
        # Always save to the current experiment directory
        save_cost_map_path = str(file_output_dir / cost_map_filename)
        save_visible_map_path = str(file_output_dir / visible_map_filename)
        with open(save_cost_map_path, 'wb') as f:
            pickle.dump(total_cost_map_list, f)
        with open(save_visible_map_path, 'wb') as f:
            pickle.dump(total_visible_map_list, f)
        print(f"[Stage 7] Total cost map computation: {time.time() - stage_start:.2f}s")




    stage_start = time.time()

    if not isinstance(total_cost_map_list, torch.Tensor):
        total_cost_map_list = torch.stack(total_cost_map_list)
    if not isinstance(total_visible_map_list, torch.Tensor):
        total_visible_maps = torch.stack(total_visible_map_list, dim=0).view(-1).to(device=device)

    # Visualize three cost maps
    total_cost_maps = lambda_foreground_geo * total_cost_map_list[:,0] + lambda_dinov1_region_match * total_cost_map_list[:,1] + lambda_dinov1_rendering * total_cost_map_list[:,2] # N, 3, H, W
    total_cost_maps = total_cost_maps.to(device)

    # Split data for point cloud 1 and point cloud 2
    pc1_indices = slice(0, len(img_1_static_list))
    pc2_indices = slice(len(img_1_static_list), None)

    # Create point clouds and masks for each set
    pc1_points = point_map_by_unprojection[pc1_indices].reshape(-1, 3)
    pc2_points = point_map_by_unprojection[pc2_indices].reshape(-1, 3)

    # Process each point cloud separately
    # Point cloud 1
    pc1_point_to_voxel = voxelization(torch.tensor(pc1_points[None], device=total_cost_maps.device), pc_voxel_size)[0]

    # Get cost maps for point cloud 1
    pc1_occlusion_maps = total_visible_maps[:len(img_1_static_list) * H * W]
    pc1_cost_maps = total_cost_maps.view(-1)[:len(img_1_static_list) * H * W][pc1_occlusion_maps]
    pc1_voxel_indices = pc1_point_to_voxel[pc1_occlusion_maps]
    pc1_cost_maps_voxel = scatter_mean(pc1_cost_maps, pc1_voxel_indices, dim=0)
    pc1_cost_maps_voxel_ori = torch.zeros(pc1_point_to_voxel.unique().max()+1, device=total_cost_maps.device)

    # Assign voxel costs for point cloud 1
    for voxel_id in pc1_voxel_indices.unique():
        pc1_cost_maps_voxel_ori[voxel_id] = pc1_cost_maps_voxel[voxel_id]
    pc1_total_cost_maps = pc1_cost_maps_voxel_ori[pc1_point_to_voxel]

    # Point cloud 2
    pc2_point_to_voxel = voxelization(torch.tensor(pc2_points[None], device=total_cost_maps.device), pc_voxel_size)[0]

    # Get cost maps for point cloud 2
    pc2_occlusion_maps = total_visible_maps[len(img_1_static_list) * H * W:]
    pc2_cost_maps = total_cost_maps.view(-1)[len(img_1_static_list) * H * W:][pc2_occlusion_maps]
    pc2_voxel_indices = pc2_point_to_voxel[pc2_occlusion_maps]
    pc2_cost_maps_voxel = scatter_mean(pc2_cost_maps, pc2_voxel_indices, dim=0)
    pc2_cost_maps_voxel_ori = torch.zeros(pc2_point_to_voxel.unique().max()+1, device=total_cost_maps.device)

    # Assign voxel costs for point cloud 2
    for voxel_id in pc2_voxel_indices.unique():
        pc2_cost_maps_voxel_ori[voxel_id] = pc2_cost_maps_voxel[voxel_id]
    pc2_total_cost_maps = pc2_cost_maps_voxel_ori[pc2_point_to_voxel]
    print(f"[Stage 8] Voxelization and cost aggregation: {time.time() - stage_start:.2f}s")

    stage_start = time.time()
    dinov1_feat_list_1 = torch.zeros(img_1_length, H, W, appearance_feature_dim).to(device)
    for index in range(len(img_1_static_list)):
        dinov1_feat_list_1[index] = predict_dinov3_feat(images[0][index].permute(1,2,0).cpu().numpy(), dinov3_vith16plus).permute(1, 2, 0)
    dinov1_feat_list_1 = dinov1_feat_list_1.view(-1, appearance_feature_dim)

    print(f"DINOv3 feature extraction and voxel aggregation: {time.time() - stage_start:.2f}s")

    pc1_dino_feat = scatter_mean(dinov1_feat_list_1.cpu(), pc1_point_to_voxel.cpu(), dim=0).cuda()
    dinov1_feat_list_1 = [] # Free memory

    dinov1_feat_list_2 = torch.zeros(img_2_length, H, W, appearance_feature_dim).to(device)
    for index in range(len(img_2_static_list)):
        dinov1_feat_list_2[index] = predict_dinov3_feat(images[0][img_1_length+index].permute(1,2,0).cpu().numpy(), dinov3_vith16plus).permute(1, 2, 0)
    dinov1_feat_list_2 = dinov1_feat_list_2.view(-1, appearance_feature_dim)

    pc2_dino_feat = scatter_mean(dinov1_feat_list_2.cpu(), pc2_point_to_voxel.cpu(), dim=0).cuda()
    dinov1_feat_list_2 = [] # Free memory

    del dinov3_vith16plus
    torch.cuda.empty_cache()
    print(f"[Stage 9] DINOv3 feature extraction and voxel aggregation: {time.time() - stage_start:.2f}s")

    stage_start = time.time()
    # Combine results for visualization (for the merged view)
    total_cost_maps = torch.cat([pc1_total_cost_maps, pc2_total_cost_maps])
    total_cost_maps_vis = apply_colormap(total_cost_maps)
    total_cost_maps_vis = total_cost_maps_vis.reshape(len(file_list), H, W, 3)

    # Store separate cost maps for individual point clouds
    pc1_cost_maps_vis = apply_colormap(pc1_total_cost_maps)
    pc1_cost_maps_vis = pc1_cost_maps_vis.reshape(len(img_1_static_list), H, W, 3)

    pc2_cost_maps_vis = apply_colormap(pc2_total_cost_maps)
    pc2_cost_maps_vis = pc2_cost_maps_vis.reshape(len(file_list) - len(img_1_static_list), H, W, 3)

    output_cost_pc_file_name_1 = str(Path(output_dir) / Path(file_list[0]).parents[1].stem / 'total_cost_maps_1_voxel.ply')
    output_cost_pc_file_name_2 = str(Path(output_dir) / Path(file_list[-1]).parents[1].stem / 'total_cost_maps_2_voxel.ply')
    output_rgb_pc_file_name_1 = str(Path(output_dir) / Path(file_list[0]).parents[1].stem / 'rgb_pc_1.ply')
    output_rgb_pc_file_name_2 = str(Path(output_dir) / Path(file_list[-1]).parents[1].stem / 'rgb_pc_2.ply')
    output_file_name_merged = str(Path(output_dir) / Path(file_list[0]).parents[1].stem / 'total_cost_maps_merged.ply')
    
    if args.vis_pc:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_map_by_unprojection.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(total_cost_maps_vis.reshape(-1, 3)) 
        o3d.io.write_point_cloud(output_file_name_merged, pcd)
        del pcd

        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(point_map_by_unprojection[:len(img_1_static_list)].reshape(-1, 3))
        pcd_1.colors = o3d.utility.Vector3dVector(total_cost_maps_vis[:len(img_1_static_list)].reshape(-1, 3))
        o3d.io.write_point_cloud(output_cost_pc_file_name_1, pcd_1)
        del pcd_1

        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(point_map_by_unprojection[len(img_1_static_list):].reshape(-1, 3))
        pcd_2.colors = o3d.utility.Vector3dVector(total_cost_maps_vis[len(img_1_static_list):].reshape(-1, 3))
        o3d.io.write_point_cloud(output_cost_pc_file_name_2, pcd_2)
        del pcd_2

        pcd_color_1 = o3d.geometry.PointCloud()
        pcd_color_1.points = o3d.utility.Vector3dVector(point_map_by_unprojection[:len(img_1_static_list)].reshape(-1, 3))
        pcd_color_1.colors = o3d.utility.Vector3dVector(images[0].permute(0, 2, 3, 1)[:len(img_1_static_list)].reshape(-1, 3).cpu().numpy())
        o3d.io.write_point_cloud(output_rgb_pc_file_name_1, pcd_color_1)
        del pcd_color_1

        pcd_color_2 = o3d.geometry.PointCloud()
        pcd_color_2.points = o3d.utility.Vector3dVector(point_map_by_unprojection[len(img_1_static_list):].reshape(-1, 3))
        pcd_color_2.colors = o3d.utility.Vector3dVector(images[0].permute(0, 2, 3, 1)[len(img_1_static_list):].reshape(-1, 3).cpu().numpy())
        o3d.io.write_point_cloud(output_rgb_pc_file_name_2, pcd_color_2)
        del pcd_color_2
        print(f"[Stage 10] Point cloud visualization saving: {time.time() - stage_start:.2f}s")
        
    stage_start = time.time()
    # Visualize the points 
    total_cluster_cost_maps_1 = []
    for index in range(img_1_length):
        cluster_cost_maps_1 = torch.zeros(H, W).to(device)
        sam_mask_1 = masks_list[index]      
        cost_map_1 = pc1_total_cost_maps.view(-1, H, W)[index].cuda()

        sam_mask_1_merged = torch.tensor(sam_mask_1).cuda()
        for i in sam_mask_1_merged.unique():
            if i == -1:
                continue
            mask_i = sam_mask_1_merged == i
            visible_mask_i = total_visible_maps.view(-1, H, W)[index].cuda()
            if args.mean_over_visible_region_during_cost_aggregation:
                cluster_cost_maps_1[mask_i] = cost_map_1[mask_i & visible_mask_i].mean() if (visible_mask_i & mask_i).sum() / mask_i.sum() > 0.3 else 0
            else:
                cluster_cost_maps_1[mask_i] = cost_map_1[mask_i].mean()
        total_cluster_cost_maps_1.append(cluster_cost_maps_1)
    
    total_cluster_cost_maps_1 = torch.stack(total_cluster_cost_maps_1)

    total_cluster_cost_maps_2 = []
    for index in range(img_2_length):
        cluster_cost_maps_2 = torch.zeros(H, W).to(device)
        sam_mask_2 = masks_list[index + img_1_length]

        sam_mask_2_merged = torch.tensor(sam_mask_2).cuda()
        cost_map_2 = pc2_total_cost_maps.view(-1, H, W)[index].cuda()
        for i in sam_mask_2_merged.unique():
            if i == -1:
                continue
            mask_i = sam_mask_2_merged == i
            visible_mask_i = total_visible_maps.view(-1, H, W)[index + img_1_length].cuda()
            if args.mean_over_visible_region_during_cost_aggregation:
                cluster_cost_maps_2[mask_i] =  cost_map_2[mask_i & visible_mask_i].mean() if (visible_mask_i & mask_i).sum() / mask_i.sum() > 0.3 else 0
            else:
                cluster_cost_maps_2[mask_i] =  cost_map_2[mask_i].mean()
        total_cluster_cost_maps_2.append(cluster_cost_maps_2)
    
    total_cluster_cost_maps_2 = torch.stack(total_cluster_cost_maps_2)

    pc1_point_to_voxel = pc1_point_to_voxel.cuda()
    pc2_point_to_voxel = pc2_point_to_voxel.cuda()

    total_cluster_cost_maps_1_visible = total_cluster_cost_maps_1[pc1_occlusion_maps.view(-1, H, W).cuda()]
    total_cluster_cost_maps_2_visible = total_cluster_cost_maps_2[pc2_occlusion_maps.view(-1, H, W).cuda()]

    if args.merge_pc_before_and_after:
        total_cluster_cost_maps_visible_for_otsu = torch.cat([total_cluster_cost_maps_1_visible, total_cluster_cost_maps_2_visible], dim=0).view(-1)
        if args.threshold_method == 'otsu':
            object_detection_threshold = threshold_otsu(total_cluster_cost_maps_visible_for_otsu[total_cluster_cost_maps_visible_for_otsu > torch.quantile(total_cluster_cost_maps_visible_for_otsu, 0.6)].view(-1)[None].cpu().numpy())
        elif args.threshold_method == 'max_entropy':
            object_detection_threshold = threshold_maximum_entropy(total_cluster_cost_maps_visible_for_otsu)
        else:
            raise ValueError(f"Invalid threshold method: {args.threshold_method}")
        object_detection_threshold_1 = object_detection_threshold
        object_detection_threshold_2 = object_detection_threshold
    else:    
        if args.threshold_method == 'otsu':
            object_detection_threshold_1 = threshold_otsu(total_cluster_cost_maps_1_visible[total_cluster_cost_maps_1_visible > torch.quantile(total_cluster_cost_maps_1_visible, args.filter_percentage_before_threshold)].view(-1)[None].cpu().numpy())
            object_detection_threshold_2 = threshold_otsu(total_cluster_cost_maps_2_visible[total_cluster_cost_maps_2_visible > torch.quantile(total_cluster_cost_maps_2_visible, args.filter_percentage_before_threshold)].view(-1)[None].cpu().numpy())
        elif args.threshold_method == 'max_entropy':
            object_detection_threshold_1 = threshold_maximum_entropy(total_cluster_cost_maps_1_visible[total_cluster_cost_maps_1_visible > torch.quantile(total_cluster_cost_maps_1_visible, args.filter_percentage_before_threshold)])
            object_detection_threshold_2 = threshold_maximum_entropy(total_cluster_cost_maps_2_visible[total_cluster_cost_maps_2_visible > torch.quantile(total_cluster_cost_maps_2_visible, args.filter_percentage_before_threshold)])
        else:
            raise ValueError(f"Invalid threshold method: {args.threshold_method}")
    object_points_mask_1 = total_cluster_cost_maps_1 > object_detection_threshold_1
    object_points_mask_2 = total_cluster_cost_maps_2 > object_detection_threshold_2

    pc_1_voxel_location = scatter_mean(torch.tensor(pc1_points, device=device), pc1_point_to_voxel, dim=0)
    pc_2_voxel_location = scatter_mean(torch.tensor(pc2_points, device=device), pc2_point_to_voxel, dim=0)

    # pc_1_voxel_location = (pc_1_voxel_location / pc_voxel_size).round()
    # pc_2_voxel_location = (pc_2_voxel_location / pc_voxel_size).round()

    mask_index_1 = -1 * torch.ones_like(object_points_mask_1).int()
    mask_index_2 = -1 * torch.ones_like(object_points_mask_2).int()

    object_list_1 = []
    mask_index_counter = 0
    for index in range(img_1_length):
        object_image_list = []
        sam_mask_1 = masks_list[index]      

        sam_mask_1_merged = torch.tensor(sam_mask_1).cuda()
        sam_mask_1_merged[~object_points_mask_1[index]] = -1

        for i in sam_mask_1_merged.unique():
            if i == -1:
                continue
            mask_i = sam_mask_1_merged == i
            if mask_i.sum() < args.min_detection_pixel:
                continue
            pc_i_indices = pc1_point_to_voxel.view(-1, H, W)[index][mask_i]
            pc_i_points = pc_1_voxel_location[pc_i_indices].unique(dim=0).view(-1, 3) # unique points of the voxel for down-sampling
            object_dict = {}
            object_dict['pc'] = pc_i_points
            object_dict['dino_feat'] = pc1_dino_feat[pc_i_indices.unique()].mean(dim=0)
            # object_dict['cost'] = total_cluster_cost_maps_1[index][mask_i.view(-1)].mean()
            # rle = mask_to_rle_pytorch(mask_i.unsqueeze(0))[0]
            # rle = coco_encode_rle(rle)
            # object_dict['frame_index'] = rle  # Store the RLE mask
            object_image_list.append(object_dict)
            mask_index_1[index][mask_i] = mask_index_counter
            mask_index_counter += 1
        object_list_1.append(object_image_list)

    object_list_2 = []
    mask_index_counter = 0
    for index in range(img_2_length):
        object_image_list = []
        sam_mask_2 = masks_list[index + img_1_length]
        sam_mask_2_merged = torch.tensor(sam_mask_2).cuda()
        sam_mask_2_merged[~object_points_mask_2[index]] = -1

        for i in sam_mask_2_merged.unique():
            if i == -1:
                continue
            mask_i = sam_mask_2_merged == i
            if mask_i.sum() < args.min_detection_pixel:
                continue
            pc_i_indices = pc2_point_to_voxel.view(-1, H, W)[index][mask_i]
            pc_i_points = pc_2_voxel_location[pc_i_indices].unique(dim=0).view(-1, 3) # unique points of the voxel for down-sampling
            object_dict = {}
            object_dict['pc'] = pc_i_points
            object_dict['dino_feat'] = pc2_dino_feat[pc_i_indices.unique()].mean(dim=0)
            object_image_list.append(object_dict)
            mask_index_2[index][mask_i] = mask_index_counter
            mask_index_counter += 1
            # vis_img = images[0][index + img_1_length].permute(1, 2, 0).cpu().numpy()
            # vis_img[mask_i.cpu().numpy()] = 0.5 * np.array([1, 0.5, 0.6]) + 0.5 * vis_img[mask_i.cpu().numpy()]
            # plt.imsave(f'cluster_cost_maps_2_{index}_{mask_index_counter}.png', vis_img)
        object_list_2.append(object_image_list)
    
    object_id_list_1, merged_object_list_1 = merge_objects(object_list_1, geometry_distance_threshold = args.geometry_distance_threshold_of_voxel_size * pc_voxel_size, visual_threshold_ratio = args.visual_threshold_ratio, geometry_threshold_ratio = args.geometry_threshold_ratio, general_threshold=args.general_threshold)
    object_id_list_2, merged_object_list_2 = merge_objects(object_list_2, geometry_distance_threshold = args.geometry_distance_threshold_of_voxel_size * pc_voxel_size, visual_threshold_ratio = args.visual_threshold_ratio, geometry_threshold_ratio = args.geometry_threshold_ratio, general_threshold=args.general_threshold)
    print(f"[Stage 11] Object detection and merging: {time.time() - stage_start:.2f}s")
    print(f"  Detected {len(object_id_list_1.unique())} objects in video 1, {len(object_id_list_2.unique())} objects in video 2")

    stage_start = time.time()
    filter_percentage_before_threshold = args.filter_percentage_before_threshold   
    object_sim_matrix = torch.zeros(len(merged_object_list_1), len(merged_object_list_2))
    for index_1, object_1 in enumerate(merged_object_list_1):
        for index_2, object_2 in enumerate(merged_object_list_2):
            object_sim_matrix[index_1, index_2] = torch.cosine_similarity(object_1['dino_feat'], object_2['dino_feat'], dim=0)

    # Get matplotlib's tab10 colormap
    cmap1 = plt.cm.tab20
    cmap2 = plt.cm.tab20b
    cmap3 = plt.cm.tab20c  # Adding a third colormap with 20 more colors
    
    # Create tensors from each colormap (taking only RGB, dropping alpha)
    colors1 = torch.tensor([cmap1(i)[:3] for i in range(20)]).float().to(device)
    colors2 = torch.tensor([cmap2(i)[:3] for i in range(20)]).float().to(device)
    colors3 = torch.tensor([cmap3(i)[:3] for i in range(20)]).float().to(device)

    
    colors = torch.cat([colors1, colors2, colors3], dim=0)
    
    vis_pc_1 = images[0][:img_1_length].permute(0, 2, 3, 1).reshape(-1, 3)
    # vis_pc_1 = torch.zeros(pc1_points.shape[0], 3).to(device)
    vis_pc_2 = images[0][img_1_length:].permute(0, 2, 3, 1).reshape(-1, 3)
    # vis_pc_2 = torch.zeros(pc2_points.shape[0], 3).to(device)
    # Save the mask of each object
    object_masks = {}
    object_masks['H'] = H
    object_masks['W'] = W
    max_score = args.max_score


    # Give each object a unit id
    unit_ids = {}
    unit_ids['video_1'] = {}
    unit_ids['video_2'] = {}
    next_id = 0
    
    # First assign ids to objects in set 1
    for obj_id in object_id_list_1.unique():
        unit_ids['video_1'][obj_id.item()] = next_id
        next_id += 1
        
    # Then assign ids to objects in set 2, reusing ids from set 1 if objects are associated
    for obj_id in object_id_list_2.unique():
        if object_sim_matrix.numel() == 0:
            max_sim = 0
        else:
            similarity_scores = object_sim_matrix[:, obj_id]
            max_sim = similarity_scores.max()
        if max_sim > args.object_similarity_threshold:
            # Object is associated with an object in set 1, use same id
            set1_obj_id = object_id_list_1.unique()[similarity_scores.argmax()]
            unit_ids['video_2'][obj_id.item()] = unit_ids['video_1'][set1_obj_id.item()]
        else:
            # New unassociated object, assign new id
            unit_ids['video_2'][obj_id.item()] = next_id
            next_id += 1

    for index in range(len(object_id_list_1.unique())):
        object_id = object_id_list_1.unique()[index]
        object_id_index = torch.where(object_id_list_1 == object_id)[0]
        unified_object_id = unit_ids['video_1'][object_id.item()]
        object_color_1 = colors[unified_object_id % colors.shape[0]]

        if unified_object_id not in object_masks:
            object_masks[unified_object_id] = {}

        for single_object_index in object_id_index:
            single_object_mask = mask_index_1 == single_object_index
            vis_pc_1[single_object_mask.view(-1)] = object_color_1

            for frame_idx in range(img_1_length):
                obj_mask = (single_object_mask.view(img_1_length, -1)[frame_idx]).view(H, W)
                if obj_mask.any():
                    if 'video_1' not in object_masks[unified_object_id]:
                        object_masks[unified_object_id]['video_1'] = {}
                    # Convert to RLE format
                    if frame_idx in object_masks[unified_object_id]['video_1']:
                        old_obj_mask = torch.tensor(mask_utils.decode(object_masks[unified_object_id]['video_1'][frame_idx]['mask'])).cuda()
                        obj_mask = obj_mask | old_obj_mask

                    rle = mask_to_rle_pytorch(obj_mask.unsqueeze(0))[0]
                    rle = coco_encode_rle(rle)
                    object_masks[unified_object_id]['video_1'][frame_idx] = {}
                    object_masks[unified_object_id]['video_1'][frame_idx]['mask'] = rle
                    object_masks[unified_object_id]['video_1'][frame_idx]['mask_size'] = obj_mask.sum().item()
                    object_cost = total_cluster_cost_maps_1[frame_idx][obj_mask.bool()]
                    object_masks[unified_object_id]['video_1'][frame_idx]['cost'] = object_cost.mean().item() / max_score
                
                    # Also save the rgb image covered with the object mask with color
                    rgb_image = images[0][frame_idx].permute(1, 2, 0).cpu().numpy()
                    
                    rgb_image[obj_mask.cpu().numpy().astype(bool)] = object_color_1.cpu().numpy()
                    vis_dir = file_output_dir / 'video_1_detection'
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_img_path = vis_dir / f'video_1_rgb_image_{frame_idx}_{object_id}.png'
                    plt.imsave(vis_img_path, rgb_image)
    
    for index in range(len(object_id_list_2.unique())):
        object_id = object_id_list_2.unique()[index]
        object_id_index = torch.where(object_id_list_2 == object_id)[0]
        unified_object_id = unit_ids['video_2'][object_id.item()]
        object_color_2 = colors[unified_object_id % colors.shape[0]]
        if unified_object_id not in object_masks:
            object_masks[unified_object_id] = {}

        for single_object_index in object_id_index:
            single_object_mask = mask_index_2 == single_object_index
            vis_pc_2[single_object_mask.view(-1)] = object_color_2

            for frame_idx in range(img_2_length):
                obj_mask = (single_object_mask.view(img_2_length, -1)[frame_idx]).view(H, W)
                if obj_mask.any():
                    if 'video_2' not in object_masks[unified_object_id]:
                        object_masks[unified_object_id]['video_2'] = {}
                    # Convert to RLE format
                    if frame_idx in object_masks[unified_object_id]['video_2']:
                        old_obj_mask = torch.tensor(mask_utils.decode(object_masks[unified_object_id]['video_2'][frame_idx]['mask'])).cuda()
                        obj_mask = obj_mask | old_obj_mask
                    rle = mask_to_rle_pytorch(obj_mask.unsqueeze(0))[0]
                    rle = coco_encode_rle(rle)
                    object_masks[unified_object_id]['video_2'][frame_idx] = {}
                    object_masks[unified_object_id]['video_2'][frame_idx]['mask'] = rle
                    object_masks[unified_object_id]['video_2'][frame_idx]['mask_size'] = obj_mask.sum().item()
                    object_cost = total_cluster_cost_maps_2[frame_idx][obj_mask.bool()]
                    object_masks[unified_object_id]['video_2'][frame_idx]['cost'] = object_cost.mean().item() / max_score

                    # Also save the rgb image covered with the object mask with color
                    rgb_image = images[0][frame_idx + img_1_length].permute(1, 2, 0).cpu().numpy()
                    rgb_image[obj_mask.cpu().numpy().astype(bool)] = object_color_2.cpu().numpy()
                    vis_dir = file_output_dir / 'video_2_detection'
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_img_path = vis_dir / f'video_2_rgb_image_{frame_idx}_{object_id}.png'
                    plt.imsave(vis_img_path, rgb_image)

    if args.vis_pc:
        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(pc1_points)
        pcd_1.colors = o3d.utility.Vector3dVector(vis_pc_1.cpu().numpy())
        # o3d.visualization.draw_geometries([pcd_1])
        # save the pcd_1
        output_merged_pcd_file_name_1 = str(Path(output_dir) / Path(file_list[0]).parents[1].stem / 'merged_pcd_instance_1.ply')
        o3d.io.write_point_cloud(output_merged_pcd_file_name_1, pcd_1)

        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(pc2_points)
        pcd_2.colors = o3d.utility.Vector3dVector(vis_pc_2.cpu().numpy())
        # o3d.visualization.draw_geometries([pcd_2])
        # save the pcd_2
        output_merged_pcd_file_name_2 = str(Path(output_dir) / Path(file_list[0]).parents[1].stem / 'merged_pcd_instance_2.ply')
        o3d.io.write_point_cloud(output_merged_pcd_file_name_2, pcd_2)

    # Save object masks dictionary
    output_masks_file = file_output_dir / 'object_masks.pkl'
    with open(output_masks_file, 'wb') as f:
        pickle.dump(object_masks, f)
    print(f"[Stage 12] Object mask generation and saving: {time.time() - stage_start:.2f}s")

    stage_start = time.time()
    # Save Ground Truth Visualization
    gt_file_path = Path(args.gt_dir) / output_name / 'segments.pkl'
    gt = pickle.load(open(gt_file_path, 'rb'))
    gt_video_1_objects = gt['video1_objects']
    gt_video_2_objects = gt['video2_objects']
    gt_meta = gt.get('objects', {})
        
    # Process ground truth objects
    for object_info in gt_meta:
        obj_k = object_info['original_obj_idx']
        in_video1 = object_info.get('in_video1', False)
        in_video2 = object_info.get('in_video2', False)
        
        # Process video 1 objects
        if in_video1 and obj_k in gt_video_1_objects:
            for frame_index, mask in gt_video_1_objects[obj_k].items():
                frame_index = int(frame_index)
                if frame_index % args.resample_rate == 0:
                    actual_frame_index = frame_index // args.resample_rate
                    gt_mask = torch.tensor(mask_utils.decode(mask))

                    gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode='bilinear', align_corners=False)[0, 0] > 0.5
                    
                    rgb_image = images[0][actual_frame_index].permute(1, 2, 0).cpu().numpy()
                    rgb_image[gt_mask.cpu().numpy()] = colors[obj_k].cpu().numpy()
                    vis_dir = file_output_dir / 'video_1_detection'
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_img_path = vis_dir / f'gt_video_1_rgb_image_{actual_frame_index}_{obj_k}.png'
                    plt.imsave(vis_img_path, rgb_image)
        
        # Process video 2 objects
        if in_video2 and obj_k in gt_video_2_objects:
            for frame_index, mask in gt_video_2_objects[obj_k].items():
                frame_index = int(frame_index)
                if frame_index % args.resample_rate == 0:
                    actual_frame_index = frame_index // args.resample_rate
                    gt_mask = torch.tensor(mask_utils.decode(mask))
                    gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode='bilinear', align_corners=False)[0, 0] > 0.5
                    
                    rgb_image = images[0][actual_frame_index + img_1_length].permute(1, 2, 0).cpu().numpy()
                    rgb_image[gt_mask.cpu().numpy()] = colors[obj_k].cpu().numpy()
                    vis_dir = file_output_dir / 'video_2_detection'
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_img_path = vis_dir / f'gt_video_2_rgb_image_{actual_frame_index}_{obj_k}.png'
                    plt.imsave(vis_img_path, rgb_image)
    print(f"[Stage 13] Ground truth visualization: {time.time() - stage_start:.2f}s")

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total processing time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"{'='*80}\n")
    
    return 

def process_video_to_frames(video_path, output_dir):
    """Process video into individual frames using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg', 
        '-loglevel', 'quiet',
        '-i', video_path, 
        '-q:v', '2', 
        '-r', '30', 
        '-start_number', '0',
        os.path.join(output_dir, '%05d.jpg')
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {e}")
        return False

def reorder_mask(mask):
    """
    Ensures mask indices are continuous after resizing.
    Sometimes resizing can eliminate certain indices, this function
    reorders the remaining indices to be continuous.
    
    Args:
        mask: An integer mask where each value represents a region
        
    Returns:
        A reordered mask with continuous indices
    """
    unique_values = np.unique(mask)
    if len(unique_values) == 0:
        return mask
        
    # Create a mapping from old indices to new consecutive indices
    # Preserve -1 values by handling them specially
    mapping = {}
    new_idx = 0
    for old_val in unique_values:
        if old_val == -1:
            mapping[old_val] = -1  # Keep -1 as -1
        else:
            mapping[old_val] = new_idx
            new_idx += 1
    
    # Create a new mask with remapped values
    reordered_mask = np.zeros_like(mask)
    for old_val, new_val in mapping.items():
        reordered_mask[mask == old_val] = new_val
        
    return reordered_mask

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_1_dir", nargs='+', default=['derek_working_room_1'])
    parser.add_argument("--video_2_dir", nargs='+', default=['derek_working_room_2'])
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--resample_rate", type=int, default=30) # 1 image per second
    parser.add_argument("--max_score", type=float, default=0.5)
    parser.add_argument("--visible_percentage", type=float, default=0.5)
    # Topk
    # norm_to_ori_range
    parser.add_argument("--norm_to_ori_range", type=eval, default=False)
    parser.add_argument("--only_left", type=eval, default=False)
    parser.add_argument('--gt_dir', type=str, default='/home/yuqun/research/scene_change/sam2_labeling/results_valid', 
                    help='Directory containing ground truth data')
    parser.add_argument("--min_detection_pixel", type=int, default=100)
    parser.add_argument("--occlusion_threshold", type=float, default=-0.02)
    parser.add_argument("--lambda_geo", type=float, default=1.0)
    parser.add_argument("--lambda_dino", type=float, default=0.5)
    parser.add_argument("--lambda_dino_region_match", type=float, default=0.5)
    parser.add_argument("--otsu_threshold", type=float, default=0.6)
    parser.add_argument("--mean_over_non_occluded_region", type=eval, default=True)
    parser.add_argument("--mean_over_visible_region_during_cost_aggregation", type=eval, default=True)
    parser.add_argument("--geometry_distance_threshold_of_voxel_size", type=int, default=2)
    parser.add_argument("--visual_threshold_ratio", type=float, default=0.7)
    parser.add_argument("--geometry_threshold_ratio", type=float, default=0.5)
    parser.add_argument("--general_threshold", type=float, default=1.4)
    parser.add_argument("--object_similarity_threshold", type=float, default=0.7)
    parser.add_argument("--use_cache", type=eval, default=False)
    parser.add_argument("--use_non_occluded_map", type=eval, default=True)
    parser.add_argument("--use_view_coverage_map", type=eval, default=True)
    parser.add_argument("--merge_pc_before_and_after", type=eval, default=True)
    parser.add_argument("--filter_percentage_before_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_method", type=str, default='max_entropy', choices=['otsu', 'max_entropy'])
    parser.add_argument("--splits", type=str, default='all', choices=['val', 'test', 'all'])
    parser.add_argument("--sets", type=str, default='All', choices=['Diverse', 'Kitchen', 'All'])
    parser.add_argument("--max_detected_objects", type=int, default=1000) # Almost infinite
    parser.add_argument("--max_detected_objects_threshold", type=float, default=0.8)  # 0.8 is typically high enough
    parser.add_argument("--vis_pc", type=eval, default=False)

    args = parser.parse_args()

    print(f"Running Job: {args.output_dir}")

    # Initialize model inside the function to ensure it's available
    device = torch.device("cuda")
    dtype = torch.bfloat16  # or torch.float16
    # model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    all_scene_dir = sorted(Path(args.gt_dir).iterdir())
    valid_scene_name = []
    finished_scene_name = []

    val_split_path = 'splits/val_split.json'
    val_split = json.load(open(val_split_path, 'r'))
    val_split['All'] = val_split['Diverse'] + val_split['Kitchen'] # Add them together for more convenient reference later  
    test_split_path = 'splits/test_split.json'
    test_split = json.load(open(test_split_path, 'r'))
    test_split['All'] = test_split['Diverse'] + test_split['Kitchen']

    if args.splits == 'val':
        valid_scene_name = val_split[args.sets]
    elif args.splits == 'test':
        valid_scene_name = test_split[args.sets]
    elif args.splits == 'all':
        valid_scene_name = val_split[args.sets] + test_split[args.sets]
    else:
        raise ValueError(f"Invalid splits: {args.splits}")
    

    print(f"Using Splits: {args.splits} and Sets: {args.sets}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    

    for scene_dir in all_scene_dir: # The already predicted one can be f'{scene_dir}_timestamp' or just f'{scene_dir}'
        for sorted_output_dir in output_dir.glob(f'{scene_dir.name}*'):
            if sorted_output_dir.is_dir():
                if Path(sorted_output_dir / 'object_masks.pkl').exists():
                    print(f'{str(sorted_output_dir)} already predicted')
                    valid_scene_name.remove(str(sorted_output_dir.name)) # Remove the scene from the valid scene list
                    finished_scene_name.append(str(sorted_output_dir.name))
                    break
    try:
        valid_scene_name.remove('IMG_5080_IMG_5081')  # Because there's a bug for this scene
    except:
        pass
    valid_scene_name = ['dynamic_1']
    print(f'Valid scene name: {valid_scene_name}')
    print(f'Total scene number: {len(valid_scene_name)}')
    print(f'Finished Scene Number: {len(finished_scene_name)}')
    # import ipdb; ipdb.set_trace()

    
    for index, video_dir_name in enumerate(valid_scene_name):
        # try:
        video_dir = Path(args.gt_dir) / video_dir_name
        if not video_dir.exists():
            continue

        video_1_dir = video_dir / 'video1_frames'
        video_2_dir = video_dir / 'video2_frames'

        # Now the names are original_video_1.extension and original_video_2.extension
        video_files = sorted([x for x in os.listdir(video_dir) if x.endswith(".mp4") or x.endswith(".mov") or x.endswith(".avi") or x.endswith(".MP4") or x.endswith(".MOV") or x.endswith(".AVI")])

        single_video_dir_1, single_video_dir_2 = video_files[0], video_files[1]

        video_1_path = video_dir / f'{single_video_dir_1}'
        video_2_path = video_dir / f'{single_video_dir_2}'

        if not video_1_dir.exists() or not video_2_dir.exists():
            # Extract the video frames from the video
            process_video_to_frames(str(video_1_path), str(video_1_dir))
            process_video_to_frames(str(video_2_path), str(video_2_dir))

        img_1_static_list = sorted(video_1_dir.glob('*.jpg'))[::args.resample_rate]
        img_2_static_list = sorted(video_2_dir.glob('*.jpg'))[::args.resample_rate]
        img_1_length = len(img_1_static_list)
        img_2_length = len(img_2_static_list)
        file_list = img_1_static_list + img_2_static_list

        print(img_1_static_list, img_2_static_list)
        predict_bbox(file_list, img_1_length, img_2_length, args.output_dir, args.norm_to_ori_range, args.visible_percentage, args, model)
        # except Exception as e:
        #     print(f"Error processing video: {e}")
        #     continue
