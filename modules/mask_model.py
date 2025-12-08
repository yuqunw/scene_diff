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

from utils import reorder_mask, remove_small_masks, masks_update
import sys
import os

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

    def predict_mask(self, image, mask_generator, min_size=100):
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
    
        
        masks_m, masks_l = masks
        masks_m, masks_l = masks_update(masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
        
        # Assign region IDs
        group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
        group_counter = 0
        
        for mask_level in (masks_l, masks_m):
            if (group_ids == -1).sum() < min_size:
                break
            
            mask_level = sorted(mask_level, key=lambda x: x['area'], reverse=True)
            
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
        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
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
            
            print(f"[MaskModel] Loaded {len(masks_list)} masks from cache")
            return masks_list
        
        # Generate new masks
        print(f"[MaskModel] Generating masks for {len(file_list)} images...")
        
        if self.mask_generator is None:
            self._initialize_sam()
        
        masks_list = []
        for idx, filepath in enumerate(file_list):
            image = np.array(Image.open(filepath)).astype(np.uint8)
            mask = self.predict_mask(image, self.mask_generator)
            mask = cv2.resize(
                mask.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
            
            # Remove small masks
            mask = remove_small_masks(mask)
            masks_list.append(reorder_mask(mask))
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(file_list)} images")
        
        # Cache the masks
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(masks_list, f)
            print(f"[MaskModel] Saved masks to cache: {cache_path}")
        
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
