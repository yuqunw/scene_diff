"""
Semantic Model Module
=====================

Handles semantic feature extraction using DINOv3 and appearance-based cost computation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from utils import (
    predict_dinov3_feat,
    reprojected_feature,
    get_dino_matched_region_cost,
)


class SemanticModel:
    """
    Handles semantic feature extraction and appearance-based cost computation
    using DINOv3.
    """
    
    def __init__(self, config, device='cuda'):
        """
        Initialize semantic model.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        self.model = None
        self.feature_dim = config['models']['dinov3']['feature_dim']
        
        # Load DINOv3 model
        self._load_dinov3_model()
    
    def _load_dinov3_model(self):
        """Load DINOv3 model."""
        dinov3_config = self.config['models']['dinov3']
        # Get project root (go up from modules/ to project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dinov3_path = os.path.join(project_root, 'submodules/dinov3')
        model_name = dinov3_config['model_name']
        
        self.model = torch.hub.load(
            dinov3_path,
            model_name,
            source='local'
        ).to(self.device).eval()
        
        print(f"[SemanticModel] Loaded DINOv3 model: {model_name}")
    
    def extract_features(self, image_np):
        """
        Extract DINOv3 features from an image.
        
        Args:
            image_np: Image as numpy array (H, W, 3)
            
        Returns:
            Feature map (C, H, W)
        """
        return predict_dinov3_feat(image_np, self.model)
    
    def extract_features_batch(self, images_np_list):
        """
        Extract features for a batch of images.
        
        Args:
            images_np_list: List of images as numpy arrays
            
        Returns:
            List of feature maps
        """
        features = []
        for img_np in images_np_list:
            feat = self.extract_features(img_np)
            features.append(feat)
        return features
    
    
    def compute_similarity(self, feat_1, feat_2):
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            feat_1, feat_2: Feature vectors
            
        Returns:
            Similarity score
        """
        return torch.cosine_similarity(feat_1, feat_2, dim=0)
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
