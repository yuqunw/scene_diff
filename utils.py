"""
Utility Functions for Two-View Scene Analysis

This module provides utilities for:
- Image loading and preprocessing
- Depth estimation and geometry processing
- Feature extraction (DINOv1, DINOv2, DINOv3)
- Multi-view geometry operations
- Mask processing and segmentation
- Point cloud operations and voxelization
- Object detection and merging
"""

import sys
import os
import math
import subprocess
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import faiss

# Add submodules to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PI3_PATH = os.path.join(PROJECT_ROOT, 'submodules/Pi3')


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Camera Calibration and Focal Length Recovery
# ============================================================================

def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    """
    Solve focal length and depth shift jointly.
    
    Minimizes |focal * xy / (z + shift) - uv| with respect to shift and focal.
    
    Args:
        uv: Image coordinates (N, 2)
        xyz: 3D points (N, 3)
        
    Returns:
        Tuple of (optimal_shift, optimal_focal)
    """
    from scipy.optimize import least_squares
    
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    """
    Solve depth shift with known focal length.
    
    Minimizes |focal * xy / (z + shift) - uv| with respect to shift.
    
    Args:
        uv: Image coordinates (N, 2)
        xyz: 3D points (N, 3)
        focal: Known focal length
        
    Returns:
        Optimal depth shift
    """
    from scipy.optimize import least_squares
    
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, 
                             dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    """
    Generate normalized UV coordinates for view plane.
    
    UV spans from (-width/diagonal, -height/diagonal) to (width/diagonal, height/diagonal)
    where diagonal = sqrt(width^2 + height^2).
    
    Args:
        width: Image width
        height: Image height
        aspect_ratio: Width/height ratio (computed if None)
        dtype: Torch data type
        device: Torch device
        
    Returns:
        UV coordinates (H, W, 2)
    """
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, 
                      width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, 
                      height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, 
                       focal: torch.Tensor = None, 
                       downsample_size: tuple[int, int] = (64, 64)):
    """
    Recover focal length and depth shift from point map.
    
    Assumes:
    - Optical center at image center
    - Undistorted image
    - Isometric in x and y directions
    
    Args:
        points: Point map (..., H, W, 3)
        mask: Valid point mask (..., H, W)
        focal: Known focal length (if provided)
        downsample_size: Size for downsampling (faster computation)
        
    Returns:
        Tuple of (focal, shift)
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
            
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
        
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


# ============================================================================
# Image Loading and Preprocessing
# ============================================================================

def load_images_as_tensor_from_list(file_list, max_size=518):
    """
    Load images from file list and convert to uniform tensor.
    
    All images are resized to have max dimension = max_size while maintaining
    aspect ratio. Dimensions are made multiples of 14.
    
    Args:
        file_list: List of image file paths
        max_size: Maximum dimension size
        
    Returns:
        Image tensor (N, 3, H, W) normalized to [0, 1]
    """
    sources = [Image.open(file).convert('RGB') for file in file_list]

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    print(f"Found {len(sources)} images. Processing...")

    # Determine target size based on first image
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    
    # Calculate new dimensions
    if H_orig > W_orig:
        new_height = max_size
        new_width = round(W_orig * (max_size / H_orig))
        new_width = round(new_width / 14) * 14  # Multiple of 14
    else:
        new_width = max_size
        new_height = round(H_orig * (max_size / W_orig))
        new_height = round(new_height / 14) * 14  # Multiple of 14

    TARGET_W, TARGET_H = new_width, new_height
    print(f"Resizing images to: (H, W) = ({TARGET_H}, {TARGET_W})")

    # Resize and convert to tensors
    tensor_list = []
    to_tensor_transform = TF.ToTensor()
    
    for img_pil in sources:
        try:
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    return torch.stack(tensor_list, dim=0)


def process_video_to_frames(video_path, output_dir):
    """
    Extract frames from video using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg', 
        '-loglevel', 'quiet',
        '-i', video_path, 
        '-q:v', '2',  # High quality
        '-r', '30',   # 30 fps
        '-start_number', '0',
        os.path.join(output_dir, '%05d.jpg')
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {e}")
        return False


# ============================================================================
# Image Transforms
# ============================================================================

class CenterPadding(torch.nn.Module):
    """Pad image to be multiple of a given size."""
    
    def __init__(self, multiple=8):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        import itertools
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class LeastResize(torch.nn.Module):
    """Resize image to nearest multiple of a given size."""
    
    def __init__(self, multiple=8):
        super().__init__()
        self.multiple = multiple

    def _get_size(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        return new_size

    @torch.inference_mode()
    def forward(self, x):
        new_size_h = self._get_size(x.shape[-2])
        new_size_w = self._get_size(x.shape[-1])
        output = F.interpolate(x, size=(new_size_h, new_size_w), mode='bilinear', align_corners=False)
        return output


# Standard transforms for different models
transform = TF.Compose([
    lambda x: x.permute(2, 0, 1),
    lambda x: x.unsqueeze(0),
    LeastResize(),
    TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

transform_dinov2 = TF.Compose([
    lambda x: x.permute(2, 0, 1),
    lambda x: x.unsqueeze(0),
    LeastResize(multiple=14),
    TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

transform_dinov3 = TF.Compose([
    lambda x: x.permute(2, 0, 1),
    lambda x: x.unsqueeze(0),
    LeastResize(multiple=16),
    TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

transform_batch = TF.Compose([
    CenterPadding(),
    TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

transform_batch_resize = TF.Compose([
    LeastResize(),
    TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# ============================================================================
# Feature Extraction
# ============================================================================

def predict_dinov1_feat(img_np, dinov1_vitb8):
    """
    Extract DINOv1 features from image.
    
    Args:
        img_np: Image as numpy array (H, W, 3)
        dinov1_vitb8: DINOv1 model
        
    Returns:
        Features (768, H, W)
    """
    ori_H, ori_W = img_np.shape[:2]
    with torch.no_grad():
        features = dinov1_vitb8.get_intermediate_layers(
            transform(torch.tensor(img_np)).to(DEVICE), 1
        )[0]
    
    H_token, W_token = math.ceil(ori_H / 8), math.ceil(ori_W / 8)
    features = features[:, 1:]  # Drop cls token
    features = features.reshape(1, H_token, W_token, 768)
    features = F.interpolate(
        features.permute(0, 3, 1, 2),
        size=(ori_H, ori_W),
        mode="bilinear",
        align_corners=False
    )[0]
    
    return features


def predict_dinov2_feat(img_np, dinov2_vitb14):
    """
    Extract DINOv2 features from image.
    
    Args:
        img_np: Image as numpy array (H, W, 3)
        dinov2_vitb14: DINOv2 model
        
    Returns:
        Features (1024, H, W)
    """
    ori_H, ori_W = img_np.shape[:2]
    
    with torch.no_grad():
        features = dinov2_vitb14.forward_features(
            transform_dinov2(torch.tensor(img_np)).to(DEVICE)
        )
    
    H_token, W_token = math.ceil(ori_H / 14), math.ceil(ori_W / 14)
    features = features["x_norm_patchtokens"].reshape(1, H_token, W_token, 1024)
    features = F.interpolate(
        features.permute(0, 3, 1, 2),
        size=(ori_H, ori_W),
        mode="bilinear",
        align_corners=False
    )[0]
    
    return features


def predict_dinov3_feat(img_np, dinov3_vith16plus):
    """
    Extract DINOv3 features from image.
    
    Args:
        img_np: Image as numpy array (H, W, 3)
        dinov3_vith16plus: DINOv3 model
        
    Returns:
        Features (1280, H, W)
    """
    ori_H, ori_W = img_np.shape[:2]
    
    with torch.no_grad():
        features = dinov3_vith16plus.forward_features(
            transform_dinov3(torch.tensor(img_np).float()).to(DEVICE)
        )
    
    H_token, W_token = math.ceil(ori_H / 16), math.ceil(ori_W / 16)
    features = features["x_norm_patchtokens"].reshape(1, H_token, W_token, 1280)
    features = F.interpolate(
        features.permute(0, 3, 1, 2),
        size=(ori_H, ori_W),
        mode="bilinear",
        align_corners=False
    )[0]
    
    return features


def predict_dinov1_feat_batch(img_np_batch, dinov1_vitb8):
    """Extract DINOv1 features for a batch of images."""
    ori_H, ori_W = img_np_batch.shape[-2:]
    
    with torch.no_grad():
        features = dinov1_vitb8.get_intermediate_layers(
            transform_batch(torch.tensor(img_np_batch)).to(DEVICE), 1
        )[0]
    
    H_token, W_token = math.ceil(ori_H / 8), math.ceil(ori_W / 8)
    features = features[:, 1:]  # Drop cls token
    features = features.reshape(-1, H_token, W_token, 768)
    features = F.interpolate(
        features.permute(0, 3, 1, 2),
        size=(H_token * 8, W_token * 8),
        mode="bilinear",
        align_corners=False
    )
    features = TF.CenterCrop((ori_H, ori_W))(features)
    return features


def predict_dinov1_feat_batch_resize(img_np_batch, dinov1_vitb8):
    """Extract DINOv1 features for a batch with resizing."""
    ori_H, ori_W = img_np_batch.shape[-2:]
    
    with torch.no_grad():
        features = dinov1_vitb8.get_intermediate_layers(
            transform_batch_resize(torch.tensor(img_np_batch)).to(DEVICE), 1
        )[0]
    
    H_token, W_token = math.ceil(ori_H / 8), math.ceil(ori_W / 8)
    features = features[:, 1:]  # Drop cls token
    features = features.reshape(-1, H_token, W_token, 768)
    features = F.interpolate(
        features.permute(0, 3, 1, 2),
        size=(ori_H, ori_W),
        mode="bilinear",
        align_corners=False
    )
    return features


# ============================================================================
# Point Cloud and Voxelization
# ============================================================================

def ravel_hash_vec(arr):
    """
    Ravel coordinates after subtracting minimum.
    
    Args:
        arr: Coordinate array (B, N, 3)
        
    Returns:
        Hashed keys (B, N)
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0].to(torch.long)
    arr_max = arr.max(1, keepdims=True)[0].to(torch.long) + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=torch.long).to(arr.device)

    # Fortran-style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys


def voxelization(xyz, voxel_size):
    """
    Voxelize point cloud.
    
    Args:
        xyz: Point cloud (B, N, 3)
        voxel_size: Voxel size (scalar)
        
    Returns:
        Point-to-voxel mapping (B, N)
    """
    B, N, _ = xyz.shape
    xyz = xyz / voxel_size
    xyz = torch.round(xyz).long()
    xyz = xyz - xyz.min(1, keepdim=True)[0]

    keys = ravel_hash_vec(xyz)

    point_to_voxel = torch.stack(
        [torch.unique(keys[b], return_inverse=True)[1] for b in range(B)], 0
    )
    return point_to_voxel


def get_robust_voxel_size(points, percentile_min=1, percentile_max=99, 
                          subsample_size=100000, scale_factor=200):
    """
    Calculate voxel size using robust percentile estimates.
    
    Args:
        points: Point cloud (N, 3)
        percentile_min: Lower percentile for robustness
        percentile_max: Upper percentile for robustness
        subsample_size: Maximum points for computation
        scale_factor: Divisor for voxel size
        
    Returns:
        Voxel size (scalar)
    """
    # Subsample if needed
    if points.shape[0] > subsample_size:
        indices = torch.randperm(points.shape[0])[:subsample_size]
        points_sample = points[indices]
    else:
        points_sample = points
    
    # Convert to numpy
    if isinstance(points_sample, torch.Tensor):
        points_np = points_sample.cpu().numpy()
    else:
        points_np = points_sample
    
    # Calculate percentiles
    min_vals = np.percentile(points_np, percentile_min, axis=0)
    max_vals = np.percentile(points_np, percentile_max, axis=0)
    
    # Maximum range across dimensions
    max_range = np.max(max_vals - min_vals, axis=0)
    
    return max_range / scale_factor


# ============================================================================
# Geometry and Projection
# ============================================================================

def get_img_coor(H, W):
    """Generate image coordinate grid."""
    y, x = torch.from_numpy(np.mgrid[:H, :W]).float()
    img_coor = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    return img_coor


def get_valid_points(depth, pose, img_coor, intrinsic):
    """
    Unproject depth map to 3D world coordinates.
    
    Args:
        depth: Depth map (1, H, W)
        pose: Camera-to-world pose (4, 4)
        img_coor: Image coordinates (H, W, 3)
        intrinsic: Camera intrinsic matrix (3, 3)
        
    Returns:
        3D points in world coordinates (H, W, 3)
    """
    K = intrinsic
    h, w = depth.shape[-2:]
    
    # Unproject to camera space
    cam_coor = img_coor @ torch.inverse(K).T * depth.permute(1, 2, 0)
    
    # Transform to world space
    ones = torch.ones_like(cam_coor[..., :1])
    h_points = torch.cat((cam_coor, ones), -1).view(-1, 4)
    world_points = (pose @ h_points.t().float()).t()[..., :3].view(h, w, 3)
    
    return world_points


def reprojected_feature(input_intrinsic, input_pose, input_depth, target_intrinsic, 
                       target_pose, target_feature, img_coors, grid_intrinsic, H=518, W=518):
    """
    Reproject features from target view to input view.
    
    Args:
        input_intrinsic: Input camera intrinsic (3, 3)
        input_pose: Input camera pose (4, 4)
        input_depth: Input depth map (1, H, W)
        target_intrinsic: Target camera intrinsic (3, 3)
        target_pose: Target camera pose (4, 4)
        target_feature: Target features (C, H, W)
        img_coors: Image coordinates (H, W, 3)
        grid_intrinsic: Grid sampling intrinsic (2, 3)
        H, W: Image dimensions
        
    Returns:
        Reprojected features (C, H, W)
    """
    # Get 3D points
    point = get_valid_points(input_depth, input_pose, img_coors, input_intrinsic).cuda()
    h_point = torch.cat((point, torch.ones_like(point[..., :1])), -1).view(-1, 4)
    
    # Transform to target camera
    h_point_neighbor = (torch.inverse(target_pose) @ h_point.t().float()).t()[..., :3].view(-1, 3)
    src_image_hpoints = (target_intrinsic @ h_point_neighbor[:, :3].t()).t()
    src_image_points = src_image_hpoints / src_image_hpoints[:, 2:]
    
    # Grid sample
    src_grid_coords = (grid_intrinsic @ src_image_points.t()).t()
    src_project_feature = F.grid_sample(
        target_feature[None], 
        src_grid_coords.view(1, H, W, 2),
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    return src_project_feature.view(-1, H, W)


def valid_mask_after_proj(input_intrinsic, input_pose, input_depth, target_intrinsic, 
                         target_pose, target_depth, img_coors, H=768, W=1024):
    """
    Compute visibility mask after projection.
    
    Args:
        input_intrinsic: Input camera intrinsic (3, 3)
        input_pose: Input camera pose (4, 4)
        input_depth: Input depth map (1, H, W)
        target_intrinsic: Target camera intrinsic (3, 3)
        target_pose: Target camera pose (4, 4)
        target_depth: Target depth map (1, H, W)
        img_coors: Image coordinates (H, W, 3)
        H, W: Image dimensions
        
    Returns:
        Visibility mask (H, W)
    """
    point = get_valid_points(input_depth, input_pose, img_coors, input_intrinsic).cuda()
    h_point = torch.cat((point, torch.ones_like(point[..., :1])), -1).view(-1, 4)
    
    # Project to target view
    h_point_neighbor = (torch.inverse(target_pose) @ h_point.t().float()).t()[..., :3].view(-1, 3)
    src_image_hpoints = (target_intrinsic @ h_point_neighbor[:, :3].t()).t()
    src_image_points = src_image_hpoints / src_image_hpoints[:, 2:]
    projected_source_depth = src_image_hpoints[:, 2]
    
    # Check bounds
    grid_mask = (src_image_points[:, 0] >= 0) & (src_image_points[:, 0] < W) & \
                (src_image_points[:, 1] >= 0) & (src_image_points[:, 1] < H) & \
                (projected_source_depth > 0)
    
    return grid_mask.view(H, W)


def foreground_distance(input_intrinsic, input_pose, input_depth, target_intrinsic, 
                       target_pose, target_depth, img_coors, grid_intrinsic, 
                       H=768, W=1024, relative=False):
    """
    Compute foreground distance (depth discrepancy after projection).
    
    Positive values indicate the point is in front of the target surface (foreground).
    
    Args:
        input_intrinsic: Input camera intrinsic (3, 3)
        input_pose: Input camera pose (4, 4)
        input_depth: Input depth map (1, H, W)
        target_intrinsic: Target camera intrinsic (3, 3)
        target_pose: Target camera pose (4, 4)
        target_depth: Target depth map (1, H, W)
        img_coors: Image coordinates (H, W, 3)
        grid_intrinsic: Grid sampling intrinsic (2, 3)
        H, W: Image dimensions
        relative: Return relative distance (normalized by depth)
        
    Returns:
        Depth discrepancy map (H, W)
    """
    point = get_valid_points(input_depth, input_pose, img_coors, input_intrinsic).cuda()
    h_point = torch.cat((point, torch.ones_like(point[..., :1])), -1).view(-1, 4)
    
    # Project to target
    h_point_neighbor = (torch.inverse(target_pose) @ h_point.t().float()).t()[..., :3].view(-1, 3)
    src_image_hpoints = (target_intrinsic @ h_point_neighbor[:, :3].t()).t()
    src_image_points = src_image_hpoints / src_image_hpoints[:, 2:]
    projected_source_depth = src_image_hpoints[:, 2]
    
    # Sample target depth
    src_grid_coords = (grid_intrinsic @ src_image_points.t()).t()
    src_proj_depths = F.grid_sample(
        target_depth.view(1, 1, H, W), 
        src_grid_coords.view(1, H, W, 2),
        mode='nearest', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    depth_dis = (src_proj_depths.view(-1) - projected_source_depth.view(-1))
    
    if relative:
        depth_dis = depth_dis / (src_proj_depths.view(-1) + 1e-6)
        depth_dis[src_proj_depths.view(-1) < 1e-6] = 0
    
    return depth_dis.view(H, W)


def calculate_mask_percentage(pose, src_pose, depth, src_depth, intrinsic, src_intrinsic, 
                              grid_intrinsic, threshold, img_coors, H=518, W=518):
    """
    Calculate what percentage of points project validly to target view.
    
    Args:
        pose: Source camera pose (4, 4)
        src_pose: Target camera pose (4, 4)
        depth: Source depth map (1, H, W)
        src_depth: Target depth map (1, H, W)
        intrinsic: Source camera intrinsic (3, 3)
        src_intrinsic: Target camera intrinsic (3, 3)
        grid_intrinsic: Grid sampling intrinsic (2, 3)
        threshold: Depth consistency threshold
        img_coors: Image coordinates (H, W, 3)
        H, W: Image dimensions
        
    Returns:
        Overlap percentage (scalar)
    """
    points = get_valid_points(depth, pose, img_coors, intrinsic).cuda()
    ones = torch.ones_like(points[..., :1])
    h_points = torch.cat((points, ones), -1).view(-1, 4)

    src_points = (src_pose.inverse() @ h_points.t()).t()
    src_image_hpoints = (src_intrinsic @ src_points[:, :3].t()).t()
    src_image_points = src_image_hpoints / src_image_hpoints[:, 2:]
    src_grid_coords = (grid_intrinsic @ src_image_points.t()).t().float()
    src_proj_depths = F.grid_sample(
        src_depth.view(1, 1, H, W), 
        src_grid_coords.view(1, H, W, 2),
        mode='nearest', 
        padding_mode='zeros', 
        align_corners=True
    )

    grid_mask = ((src_grid_coords >= -1) & (src_grid_coords <= 1)).all(-1).view(-1) & \
                (src_image_hpoints[:, 2:] > 0).view(-1)

    percentage = (src_proj_depths > 0).sum() / grid_mask.numel()

    return percentage


# ============================================================================
# Cost and Distance Computation
# ============================================================================

def compute_conf_norm_distance(foreground_dis, conf):
    """
    Compute confidence-normalized distance.
    
    Args:
        foreground_dis: Foreground distance map (H, W)
        conf: Confidence map (H, W)
        
    Returns:
        Confidence-weighted distance (H, W)
    """
    return foreground_dis * conf


def robust_filter(values, quantile=0.05):
    """
    Robustly filter and normalize values using quantiles.
    
    Args:
        values: Input values (tensor)
        quantile: Quantile for filtering
        
    Returns:
        Tuple of (filtered_values, valid_mask)
    """
    try:
        value_min = torch.quantile(values, quantile)
        value_max = torch.quantile(values, 1 - quantile)
        value_robust_min = value_min - quantile / (1 - quantile*2) * (value_max - value_min)
        value_robust_max = value_max + quantile / (1 - quantile*2) * (value_max - value_min)
        value_valid = (values > value_min) & (values < value_max)
        value_new = (values - value_min) * (value_robust_max - value_robust_min) / (value_max - value_min) + value_robust_min
    except:
        value_new = values.cuda()
        value_valid = torch.ones_like(values).bool().cuda()
    return value_new, value_valid


def compute_scale_and_shift(prediction, target, mask):
    """
    Compute scale and shift for depth alignment.
    
    Solves: target = scale * prediction + shift
    
    Args:
        prediction: Predicted depth (B, N, 1) or (N,)
        target: Target depth (B, N, 1) or (N,)
        mask: Valid mask (B, N, 1) or (N,)
        
    Returns:
        Tuple of (scale, shift)
    """
    if len(prediction.shape) == 1:
        prediction = prediction.view(1, -1, 1)
        target = target.view(1, -1, 1)
        mask = mask.view(1, -1, 1)

    # System matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # Right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # Solution: x = A^-1 . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def get_dino_matched_region_cost(dinov1_feat_1, dinov1_feat_2, sam_mask_1_merged, 
                                 sam_mask_2_merged, valid_mask_1, valid_mask_2):
    """
    Compute region matching cost using DINO features.
    
    Args:
        dinov1_feat_1: DINO features for image 1 (C, H, W)
        dinov1_feat_2: DINO features for image 2 (C, H, W)
        sam_mask_1_merged: Segmentation mask for image 1 (H, W)
        sam_mask_2_merged: Segmentation mask for image 2 (H, W)
        valid_mask_1: Valid mask for image 1 (H, W)
        valid_mask_2: Valid mask for image 2 (H, W)
        foreground_dis_1: Foreground distance for image 1 (H, W)
        foreground_dis_2: Foreground distance for image 2 (H, W)
        
    Returns:
        Tuple of (cost_1, cost_2, match_indices_1, match_indices_2, sim_1, sim_2)
    """
    # Aggregate features per region
    dinov1_region_1_tensor = torch.zeros(dinov1_feat_1.shape[0], len(sam_mask_1_merged.unique()))
    dinov1_region_2_tensor = torch.zeros(dinov1_feat_1.shape[0], len(sam_mask_2_merged.unique()))

    for i, mask_idx in enumerate(sam_mask_1_merged.unique()):
        mask_i = sam_mask_1_merged == mask_idx
        if mask_idx != -1:
            dinov1_region_1_tensor[:, i] = dinov1_feat_1[:, mask_i].mean(dim=-1)

    for i, mask_idx in enumerate(sam_mask_2_merged.unique()):
        mask_i = sam_mask_2_merged == mask_idx
        if mask_idx != -1:
            dinov1_region_2_tensor[:, i] = dinov1_feat_2[:, mask_i].mean(dim=-1)

    # Drop background region
    dinov1_region_1_tensor = dinov1_region_1_tensor[:, 1:]
    dinov1_region_2_tensor = dinov1_region_2_tensor[:, 1:]

    # Compute similarity matrix
    dinov1_region_match_sim = torch.cosine_similarity(
        dinov1_region_1_tensor[:, :, None], 
        dinov1_region_2_tensor[:, None], 
        dim=0
    )

    dinov1_region_match_sim_1, dinov1_region_match_sim_1_index = dinov1_region_match_sim.max(dim=1)
    dinov1_region_match_sim_2, dinov1_region_match_sim_2_index = dinov1_region_match_sim.max(dim=0)

    dinov1_region_match_cost_1 = torch.zeros_like(sam_mask_1_merged).float()
    dinov1_region_match_cost_2 = torch.zeros_like(sam_mask_2_merged).float()

    occlusion_threshold = 0.6

    for mask_idx in sam_mask_1_merged.unique():
        if mask_idx == -1:
            continue
        mask_i = sam_mask_1_merged == mask_idx
        if (mask_i & valid_mask_1).float().sum() > (mask_i.float().sum() * occlusion_threshold):
            dinov1_region_match_cost_1[mask_i] = 1 - dinov1_region_match_sim_1[mask_idx]
        else:
            dinov1_region_match_cost_1[mask_i] = 0

    for mask_idx in sam_mask_2_merged.unique():
        if mask_idx == -1:
            continue
        mask_i = sam_mask_2_merged == mask_idx
        if (mask_i & valid_mask_2).float().sum() > (mask_i.float().sum() * occlusion_threshold):
            dinov1_region_match_cost_2[mask_i] = 1 - dinov1_region_match_sim_2[mask_idx]
        else:
            dinov1_region_match_cost_2[mask_i] = 0

    dinov1_region_match_cost_1[~valid_mask_1] = 0
    dinov1_region_match_cost_2[~valid_mask_2] = 0
    
    return (dinov1_region_match_cost_1, dinov1_region_match_cost_2, 
            dinov1_region_match_sim_1_index, dinov1_region_match_sim_2_index,
            dinov1_region_match_sim_1, dinov1_region_match_sim_2)


# ============================================================================
# Mask Processing
# ============================================================================

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.ndim < 3:
        num_top_k = scores.shape[0]
        if keep_conf.sum() == 0:
            index = scores.topk(num_top_k).indices
            keep_conf[index] = True
        if keep_inner_u.sum() == 0:
            index = scores.topk(num_top_k).indices
            keep_inner_u[index] = True
        if keep_inner_l.sum() == 0:
            index = scores.topk(num_top_k).indices
            keep_inner_l[index] = True        
    else:
        if keep_conf.sum() == 0:
            index = scores.topk(3).indices
            keep_conf[index, 0] = True
        if keep_inner_u.sum() == 0:
            index = scores.topk(3).indices
            keep_inner_u[index, 0] = True
        if keep_inner_l.sum() == 0:
            index = scores.topk(3).indices
            keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def threshold_maximum_entropy(image, nbins=256):
    """
    Compute threshold using Maximum Entropy method (Kapur's method).
    
    Args:
        image: Input image or values (torch.Tensor or numpy array)
        nbins: Number of histogram bins
        
    Returns:
        Threshold value (float)
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
    
    image = image.flatten()
    min_val = image.min()
    max_val = image.max()
    
    # Histogram
    hist = torch.histc(image.float(), bins=nbins, min=min_val.item(), max=max_val.item())
    hist = hist.float() / hist.sum()
    
    epsilon = torch.finfo(torch.float32).eps
    max_entropy = torch.tensor(-float('inf'), device=image.device)
    best_threshold = torch.tensor(0.0, device=image.device)
    bin_width = (max_val - min_val) / nbins
    
    for t in range(1, nbins):
        P0 = hist[:t].sum()
        P1 = hist[t:].sum()
        
        if P0 == 0 or P1 == 0:
            continue
        
        # Entropy of background and foreground
        hist_bg = hist[:t] / (P0 + epsilon)
        hist_fg = hist[t:] / (P1 + epsilon)
        H0 = -(hist_bg * torch.log(hist_bg + epsilon)).sum()
        H1 = -(hist_fg * torch.log(hist_fg + epsilon)).sum()
        
        total_entropy = H0 + H1
        
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            best_threshold = min_val + t * bin_width
    
    return best_threshold.item()

def remove_small_masks(mask, min_size=100):
    """
    Remove small masks from the mask.
    
    Args:
        mask: Input mask (H, W)
        min_size: Minimum size of the mask
    """
    unique_values = np.unique(mask)
    for val in unique_values:
        if val == -1:
            continue
        mask_i = mask == val
        if mask_i.sum() < min_size:
            mask[mask_i] = -1
    return mask

def reorder_mask(mask):
    """
    Ensure mask indices are continuous after resizing.
    
    Args:
        mask: Integer mask (H, W)
        
    Returns:
        Reordered mask with continuous indices
    """
    unique_values = np.unique(mask)
    if len(unique_values) == 0:
        return mask
    
    mapping = {}
    new_idx = 0
    for old_val in unique_values:
        if old_val == -1:
            mapping[old_val] = -1  # Preserve background
        else:
            mapping[old_val] = new_idx
            new_idx += 1
    
    reordered_mask = np.zeros_like(mask)
    for old_val, new_val in mapping.items():
        reordered_mask[mask == old_val] = new_val
    
    return reordered_mask



def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encode masks to uncompressed RLE format.
    
    Args:
        tensor: Binary masks (B, H, W)
        
    Returns:
        List of RLE dictionaries
    """
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)
    
    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()
    
    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat([
            torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
            cur_idxs + 1,
            torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
        ])
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    
    return out


def coco_encode_rle(uncompressed_rle):
    """
    Convert uncompressed RLE to COCO format.
    
    Args:
        uncompressed_rle: Uncompressed RLE dictionary
        
    Returns:
        COCO RLE dictionary
    """
    from pycocotools import mask as mask_utils
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def merged_mask_with_list(mask, lists):
    """
    Merge mask regions according to provided lists.
    
    Args:
        mask: Input mask (H, W)
        lists: List of lists, each containing region IDs to merge
        
    Returns:
        Merged mask with continuous IDs
    """
    new_mask = mask.clone()
    total_list = new_mask.unique().tolist()
    
    for l in lists:
        i = l[0]
        total_list.remove(i)
        for j in range(1, len(l)):
            new_mask[new_mask == l[j]] = i
            total_list.remove(l[j])
    
    i = total_list[0]
    for j in total_list[1:]:
        new_mask[new_mask == j] = i
    
    # Ensure continuous IDs
    new_mask_unique = new_mask.unique().tolist()
    for i, m in enumerate(new_mask_unique):
        new_mask[new_mask == m] = i
    
    return new_mask


# ============================================================================
# Object Detection and Merging
# ============================================================================

def merge_objects(object_list, geometry_distance_threshold=0.5, visual_threshold_ratio=0.7, 
                 geometry_threshold_ratio=0.5, general_threshold=1.2):
    """
    Merge objects across frames using geometric and visual similarity.
    
    Args:
        object_list: List of object lists per frame
        geometry_distance_threshold: Distance threshold for geometric matching
        visual_threshold_ratio: Visual similarity threshold
        geometry_threshold_ratio: Geometric overlap threshold
        general_threshold: Combined threshold for merging
        
    Returns:
        Tuple of (object_id_list, merged_object_list)
    """
    object_id = 0
    merged_object_list = []
    merged_object_id_list = []
    first_one = True
    
    for obj_image_list in object_list:
        if len(obj_image_list) == 0:
            continue
        
        if first_one:
            for obj in obj_image_list:
                obj['num_detections'] = 1
                merged_object_list.append(obj)
                merged_object_id_list.append(object_id)
                object_id += 1
            first_one = False
            continue
        
        # Build FAISS indices for fast nearest neighbor search
        point_arrays = [obj['pc'].cpu().numpy() for obj in merged_object_list]
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
        
        geo_similarity_matrix = torch.zeros(len(obj_image_list), len(merged_object_list))
        vis_similarity_matrix = torch.zeros(len(obj_image_list), len(merged_object_list))
        
        for i, arr in enumerate(point_arrays):
            indices[i].add(arr)
        
        # Compute similarities
        for merged_idx, merged_obj in enumerate(merged_object_list):
            for obj_idx, obj in enumerate(obj_image_list):
                faiss_index = indices[merged_idx]
                obj_pc_np = obj['pc'].cpu().numpy()
                
                # Find nearest neighbors
                D, I = faiss_index.search(obj_pc_np, k=1)
                
                geometry_overlap = (D < geometry_distance_threshold ** 2).sum() / len(D)
                feature_similarity = torch.cosine_similarity(
                    merged_obj['dino_feat'], obj['dino_feat'], dim=0
                )
                
                geo_similarity_matrix[obj_idx, merged_idx] = geometry_overlap
                vis_similarity_matrix[obj_idx, merged_idx] = feature_similarity
        
        # Merge objects
        for new_obj_idx, new_obj in enumerate(obj_image_list):
            merged_similarity = geo_similarity_matrix[new_obj_idx] + vis_similarity_matrix[new_obj_idx]
            merged_similarity_index = merged_similarity.argmax()
            
            if (merged_similarity[merged_similarity_index] < general_threshold) or \
               (geo_similarity_matrix[new_obj_idx, merged_similarity_index] < geometry_threshold_ratio):
                # Create new object
                new_obj['num_detections'] = 1
                merged_object_list.append(new_obj)
                merged_object_id_list.append(object_id)
                object_id += 1
                continue
            
            # Merge with existing object
            merged_obj = merged_object_list[merged_similarity_index]
            merged_obj['pc'] = torch.cat([merged_obj['pc'], new_obj['pc']], dim=0).unique(dim=0)
            merged_obj['dino_feat'] = (merged_obj['dino_feat'] * merged_obj['num_detections'] + 
                                      new_obj['dino_feat']) / (merged_obj['num_detections'] + 1)
            # Normalize
            merged_obj['dino_feat'] = merged_obj['dino_feat'] / merged_obj['dino_feat'].norm(dim=0, keepdim=True)
            merged_obj['num_detections'] += 1
            merged_object_id_list.append(merged_similarity_index.item())

    return torch.tensor(merged_object_id_list), merged_object_list


# ============================================================================
# Visualization Utilities
# ============================================================================

def apply_colormap(cost_map):
    """
    Convert cost map to RGB using viridis colormap.
    
    Args:
        cost_map: Cost values (H, W) or (N,)
        
    Returns:
        RGB image (..., 3)
    """
    if isinstance(cost_map, torch.Tensor):
        cost_map = cost_map.cpu().numpy()
    
    # Normalize to [0, 1]
    cost_map = (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min() + 1e-8)
    
    # Apply colormap
    colored_map = cm.viridis(cost_map)
    return colored_map[..., :3]  # Drop alpha



def plot_figure(img1_np, img2_np, sam_mask_1_merged, sam_mask_2_merged, 
               attribute_pixel_1, attribute_pixel_2, attribute_region_1, 
               attribute_region_2, attribute, figure_name, output_dir):
    """
    Plot comparison figure for two-view analysis.
    
    Args:
        img1_np: Image 1 (H, W, 3)
        img2_np: Image 2 (H, W, 3)
        sam_mask_1_merged: Mask for image 1
        sam_mask_2_merged: Mask for image 2
        attribute_pixel_1: Pixel-level attribute for image 1
        attribute_pixel_2: Pixel-level attribute for image 2
        attribute_region_1: Region-level attribute for image 1
        attribute_region_2: Region-level attribute for image 2
        attribute: Attribute name (str)
        figure_name: Output filename prefix
        output_dir: Output directory
    """
    vmin = min(attribute_region_1.min().item(), attribute_region_2.min().item())
    vmax = max(attribute_region_1.max().item(), attribute_region_2.max().item())

    valid_mask_1 = attribute_region_1 != 0
    valid_mask_2 = attribute_region_2 != 0

    vis_1_pixel = (attribute_pixel_1 - vmin) * valid_mask_1
    vis_2_pixel = (attribute_pixel_2 - vmin) * valid_mask_2
    vis_1_region = (attribute_region_1 - vmin) * valid_mask_1
    vis_2_region = (attribute_region_2 - vmin) * valid_mask_2

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 4, 1)
    plt.axis('off')
    plt.imshow(img1_np)
    plt.title('Image 1')

    plt.subplot(2, 4, 2)
    plt.axis('off')
    plt.imshow(sam_mask_1_merged.cpu().numpy())
    plt.title('Image 1 Mask')

    plt.subplot(2, 4, 3)
    plt.axis('off')
    plt.imshow(vis_1_region.cpu().numpy(), vmin=0, vmax=vmax - vmin)
    plt.title(f'{attribute} Region Distance')

    plt.subplot(2, 4, 4)
    plt.axis('off')
    plt.imshow(vis_1_pixel.cpu().numpy(), vmin=0, vmax=vmax - vmin)
    plt.title(f'{attribute} Pixel Distance')

    plt.subplot(2, 4, 5)
    plt.axis('off')
    plt.imshow(img2_np)
    plt.title('Image 2')

    plt.subplot(2, 4, 6)
    plt.axis('off')
    plt.imshow(sam_mask_2_merged.cpu().numpy())
    plt.title('Image 2 Mask')

    plt.subplot(2, 4, 7)
    plt.axis('off')
    plt.imshow(vis_2_region.cpu().numpy(), vmin=0, vmax=vmax - vmin)
    plt.title(f'{attribute} Region Distance')

    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.imshow(vis_2_pixel.cpu().numpy(), vmin=0, vmax=vmax - vmin)
    plt.title(f'{attribute} Pixel Distance')

    plt.savefig(f'{output_dir}/{figure_name}_{attribute}.png')


def plot_single_img_figure(img1_np, sam_mask_1_merged, attribute_pixel_1, 
                           attribute_region_1, attribute, figure_name, output_dir):
    """Plot analysis figure for single image."""
    vmin = attribute_region_1.min().item()
    vmax = attribute_region_1.max().item()

    valid_mask_1 = attribute_region_1 != 0
    vis_1_pixel = (attribute_pixel_1 - vmin) * valid_mask_1
    vis_1_region = (attribute_region_1 - vmin) * valid_mask_1

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 4, 1)
    plt.axis('off')
    plt.imshow(img1_np)
    plt.title('Image 1')

    plt.subplot(1, 4, 2)
    plt.axis('off')
    plt.imshow(sam_mask_1_merged.cpu().numpy())
    plt.title('Image 1 Mask')

    plt.subplot(1, 4, 3)
    plt.axis('off')
    plt.imshow(vis_1_region.cpu().numpy(), vmin=0, vmax=vmax - vmin)
    plt.title(f'{attribute} Region Distance')

    plt.subplot(1, 4, 4)
    plt.axis('off')
    plt.imshow(vis_1_pixel.cpu().numpy(), vmin=0, vmax=vmax - vmin)
    plt.title(f'{attribute} Pixel Distance')

    plt.savefig(f'{output_dir}/{figure_name}_{attribute}.png')

