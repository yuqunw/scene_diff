"""
Mask Model Module
=================

Handles SAM-based segmentation mask generation and management.
"""

import os
import pickle
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path

from utils import reorder_mask, remove_small_masks, masks_update
import sys
import os

SAM_LONG_SIDE = 2000

class MaskModel:
    """
    Handles segmentation mask generation using SAM (Segment Anything Model).
    """
    
    def __init__(self, config, device='cuda'):
        """
        Initialize mask model.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.mask_generator = None
    
    def _initialize_sam(self):
        """Initialize SAM model and mask generator."""
        
        try:
            # Should work if the segment_anything_ls is installed as package
            from segment_anything_ls import build_sam, SamAutomaticMaskGenerator
        except ImportError:
            # Get project root (go up from modules/ to project root)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            segment_ls_path = os.path.join(project_root, 'submodules/segment-anything-langsplat-modified')
            if segment_ls_path not in sys.path:
                sys.path.append(segment_ls_path)
        
        sam_config = self.config['models']['sam']
        checkpoint = os.path.expandvars(sam_config['checkpoint'])
        
        self.mask_generator = SamAutomaticMaskGenerator(
            model=build_sam(checkpoint=checkpoint).to(device=self.device),
            points_per_side=sam_config['points_per_side'],
            pred_iou_thresh=sam_config['pred_iou_thresh'],
            box_nms_thresh=sam_config['box_nms_thresh'],
            stability_score_thresh=sam_config['stability_score_thresh'],
            crop_n_layers=sam_config['crop_n_layers'],
            crop_n_points_downscale_factor=sam_config['crop_n_points_downscale_factor'],
            min_mask_region_area=sam_config['min_mask_region_area'],
        )
        
        print(f"[MaskModel] Initialized SAM with checkpoint: {checkpoint}")

    def predict_mask(self, image, mask_generator, min_size=50):
        """
        Generate non-overlapping segmentation masks using SAM. 
        Follow the code of LangSplats.
        
        Args:
            image: Input image (H, W, 3) as uint8 numpy array
            mask_generator: SAM mask generator
            min_size: Minimum region size
            
        Returns:
            Segmentation mask (H, W) with integer region IDs
        """
        masks = mask_generator.generate(image)
        if isinstance(masks, tuple):
            masks_m, masks_l = masks
            masks_m, masks_l = masks_update(masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
            mask_levels = (masks_m, masks_l)
        else:
            mask_levels = (masks,)
        
        # Assign region IDs
        group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
        group_counter = 0
        
        for mask_level in mask_levels:
            if (group_ids == -1).sum() < min_size:
                break
            
            mask_level = sorted(mask_level, key=lambda x: x['area'], reverse=False) # Large to small, so that small would be on the top.
            
            for mask_dict in mask_level:
                mask_single = mask_dict["segmentation"]
                non_mask_area = (group_ids == -1)
                to_mask_area = mask_single & non_mask_area
                
                if to_mask_area.sum() < min_size:
                    continue
                
                group_ids[to_mask_area] = group_counter
                group_counter += 1
                
                if (group_ids == -1).sum() < min_size:
                    break
        
        return group_ids

    def _resize_normalize_mask(self, mask, H, W):
        if mask.shape != (H, W):
            mask = cv2.resize(
                mask.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
        return reorder_mask(mask)

    def _resize_image_for_sam(self, image, long_side=SAM_LONG_SIDE):
        h, w = image.shape[:2]
        max_side = max(h, w)
        if max_side <= 0:
            raise ValueError(f"Invalid image shape for SAM resize: {image.shape}")
        scale = float(long_side) / float(max_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if (new_h, new_w) == (h, w):
            return image
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _load_cache_as_name_map(self, path, file_names_hint=None):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if 'file_names' in data and 'masks' in data:
                return {
                    name: mask for name, mask in zip(data['file_names'], data['masks'])
                }
            # Treat plain dict[str, mask] as already named mapping.
            return data

        if isinstance(data, list):
            if file_names_hint is None or len(file_names_hint) != len(data):
                return None
            return {
                name: mask for name, mask in zip(file_names_hint, data)
            }
        return None

    def load_or_generate_masks(self, file_list, H, W, cache_path=None):
        """
        Load cached SAM masks or generate new ones.
        
        Args:
            file_list: List of image paths
            H, W: Target dimensions
            cache_path: Path to cached masks file
            
        Returns:
            List of non-overlapping masks with each value being the index of the mask
            and -1 for no mask
        """
        target_names = [Path(p).name for p in file_list]
        name_to_mask = {}
        cache_dir = None
        if cache_path:
            cache_path = Path(cache_path)
            cache_dir = cache_path.parent

            # 1) Primary aligned cache.
            if cache_path.exists():
                loaded = self._load_cache_as_name_map(cache_path, file_names_hint=target_names)
                if loaded is not None:
                    name_to_mask.update(loaded)

            # 2) Reused half+full_test cache with companion name list.
            half_path = cache_dir / "sam_masks_half_plus_full_test.pkl"
            half_names_json = cache_dir / "sam_masks_half_plus_full_test_names.json"
            if half_path.exists() and half_names_json.exists():
                try:
                    import json
                    with open(half_names_json, "r") as f:
                        half_names = json.load(f).get("file_names", [])
                    loaded = self._load_cache_as_name_map(half_path, file_names_hint=half_names)
                    if loaded is not None:
                        name_to_mask.update(loaded)
                except Exception:
                    pass

            # 3) Additional generated cache.
            additional_path = cache_dir / "sam_masks_additional.pkl"
            if additional_path.exists():
                loaded = self._load_cache_as_name_map(additional_path)
                if loaded is not None:
                    name_to_mask.update(loaded)

            # 4) Released precomputed masks (e.g. PASLCD); take precedence over local caches.
            released_path = cache_dir / "sam_masks.npz"
            if released_path.exists():
                data = np.load(released_path)
                name_to_mask.update(
                    (name, mask.astype(np.int64))
                    for name, mask in zip(data["file_names"].tolist(), data["masks"])
                )

        # Keep a raw copy for cache writing (potentially higher resolution),
        # and a normalized copy for current run consumption.
        raw_name_to_mask = {
            n: np.asarray(m) for n, m in name_to_mask.items() if n in target_names
        }
        name_to_mask = {
            n: self._resize_normalize_mask(np.asarray(m), H, W)
            for n, m in raw_name_to_mask.items()
        }

        missing_names = [n for n in target_names if n not in name_to_mask]
        if len(missing_names) == 0:
            masks_list = [name_to_mask[n] for n in target_names]
            print(f"[MaskModel] Loaded {len(masks_list)} masks from cache")
            return masks_list
        
        # Generate new masks
        print(f"[MaskModel] Generating masks for {len(file_list)} images...")
        
        if self.mask_generator is None:
            self._initialize_sam()
        
        masks_list = [name_to_mask.get(n, None) for n in target_names]
        additional_name_to_mask_high = {}
        for idx, filepath in enumerate(file_list):
            file_name = Path(filepath).name
            if file_name in name_to_mask:
                continue
            image = np.array(Image.open(filepath).convert("RGB")).astype(np.uint8)
            image = self._resize_image_for_sam(image, long_side=SAM_LONG_SIDE)
            mask_high = self.predict_mask(image, self.mask_generator)
            mask = cv2.resize(
                mask_high.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
            
            # Remove small masks
            mask = remove_small_masks(mask, min_size=50)
            mask = reorder_mask(mask)
            masks_list[idx] = mask
            additional_name_to_mask_high[file_name] = mask_high
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(file_list)} images")

        # Fill from already loaded cache where available.
        for i, n in enumerate(target_names):
            if masks_list[i] is None and n in name_to_mask:
                masks_list[i] = name_to_mask[n]

        if any(m is None for m in masks_list):
            raise RuntimeError("Some masks are still missing after cache loading and generation.")

        # Cache the masks
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Prefer higher-resolution masks when available for cache persistence.
            cache_name_to_mask = dict(raw_name_to_mask)
            cache_name_to_mask.update(additional_name_to_mask_high)
            cache_masks = [
                cache_name_to_mask.get(n, masks_list[i])
                for i, n in enumerate(target_names)
            ]
            with open(cache_path, 'wb') as f:
                pickle.dump({'file_names': target_names, 'masks': cache_masks}, f)
            print(f"[MaskModel] Saved masks to cache: {cache_path}")

            if additional_name_to_mask_high:
                additional_path = cache_path.parent / "sam_masks_additional.pkl"
                existing = {}
                if additional_path.exists():
                    loaded = self._load_cache_as_name_map(additional_path)
                    if isinstance(loaded, dict):
                        existing = loaded
                existing.update(additional_name_to_mask_high)
                with open(additional_path, 'wb') as f:
                    pickle.dump(existing, f)
                print(f"[MaskModel] Saved additional masks to cache: {additional_path}")
        
        return masks_list
    
    def aggregate_to_regions(self, pixel_values, mask):
        """
        Aggregate pixel-level values to region-level using mask.
        
        Args:
            pixel_values: Per-pixel values (H, W)
            mask: Segmentation mask (H, W) with region IDs
            
        Returns:
            Region-aggregated values (H, W)
        """
        region_values = torch.zeros_like(mask).float()
        for region_id in mask.unique():
            if region_id == -1:
                continue
            region_mask = mask == region_id
            if region_mask.sum() > 0 and pixel_values[region_mask].numel() > 0:
                region_values[region_mask] = pixel_values[region_mask].mean()
        return region_values
    
    def get_mask_statistics(self, masks_list):
        """
        Get statistics about the masks.
        
        Args:
            masks_list: List of segmentation masks
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_images': len(masks_list),
            'regions_per_image': [],
            'avg_region_size': []
        }
        
        for mask in masks_list:
            unique_regions = np.unique(mask)
            unique_regions = unique_regions[unique_regions != -1]
            stats['regions_per_image'].append(len(unique_regions))
            
            if len(unique_regions) > 0:
                sizes = [(mask == region_id).sum() for region_id in unique_regions]
                stats['avg_region_size'].append(np.mean(sizes))
        
        stats['avg_regions_per_image'] = np.mean(stats['regions_per_image'])
        stats['avg_region_size_overall'] = np.mean(stats['avg_region_size'])
        
        return stats
    
    def cleanup(self):
        """Clean up model resources."""
        if self.mask_generator is not None:
            del self.mask_generator
            self.mask_generator = None
            torch.cuda.empty_cache()
