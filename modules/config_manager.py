"""
Configuration Manager
====================

Utilities for loading and validating configuration files.
"""

import os
from pathlib import Path
import yaml


class ConfigManager:
    """
    Manages configuration loading, validation, and access.
    """
    
    DEFAULT_CONFIG = {
        'models': {
            'pi3': {'name': 'yyfz233/Pi3', 'device': 'cuda'},
            'sam': {
                'checkpoint': '${SAM_CHECKPOINT}',
                'points_per_side': 32,
                'pred_iou_thresh': 0.7,
                'box_nms_thresh': 0.7,
                'stability_score_thresh': 0.85,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 1,
                'min_mask_region_area': 100,
            },
            'dinov3': {
                'path': '${DINOV3_PATH}',
                'model_name': 'dinov3_vith16plus',
                'feature_dim': 1280,
            },
        },
        'processing': {
            'visible_percentage': 0.5,
            'subsample_size': 1000000,
            'total_voxel_number': 200,
            'occlusion_threshold': -0.02,
        },
        'costs': {
            'lambda_geo': 1.0,
            'lambda_dino': 0.3,
            'lambda_dino_region_match': 0.1,
        },
        'detection': {
            'min_detection_pixel': 100,
            'threshold_method': 'otsu',
            'change_region_threshold': None,
            'filter_percentage_before_threshold': 0.6,
            'max_score': 0.5,
            'max_detected_objects': 1000,
        },
        'merging': {
            'geometry_distance_threshold_of_voxel_size': 2,
            'visual_threshold_ratio': 0.7,
            'geometry_threshold_ratio': 0.5,
            'general_threshold': 1.4,
            'object_similarity_threshold': 0.7,
        },
        'cache': {
            'use_cache': False,
            'save_cache': False,
        },
        'visualization': {
            'vis_pc': False,
            'save_detections': True,
            'save_gt_visualization': True,
        },
        'output': {
            'output_dir': 'output',
            'save_object_masks': True,
        },
    }
    
    def __init__(self, config_path=None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path is not None:
            self.load(config_path)
        
        self._expand_env_vars()
    
    def load(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Deep merge with default config
        self.config = self._deep_merge(self.config, user_config)
    
    def _deep_merge(self, default, override):
        """
        Deep merge two dictionaries.
        
        Args:
            default: Default dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _expand_env_vars(self):
        """Expand environment variables in configuration strings."""
        def expand_recursive(obj):
            if isinstance(obj, dict):
                return {k: expand_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_recursive(item) for item in obj]
            elif isinstance(obj, str):
                return os.path.expandvars(obj)
            else:
                return obj
        
        self.config = expand_recursive(self.config)
    
    def get(self, key_path, default=None):
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'models.pi3.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """
        Set configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated key path (e.g., 'models.pi3.name')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, config_path):
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def validate(self):
        """
        Validate configuration values.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate processing parameters
        if self.get('processing.visible_percentage') <= 0 or self.get('processing.visible_percentage') > 1:
            errors.append("processing.visible_percentage must be in (0, 1]")
        
        # Validate cost weights
        for key in ['lambda_geo', 'lambda_dino', 'lambda_dino_region_match']:
            if self.get(f'costs.{key}') < 0:
                errors.append(f"costs.{key} must be non-negative")
        
        # Validate detection parameters
        if self.get('detection.min_detection_pixel') < 1:
            errors.append("detection.min_detection_pixel must be >= 1")
        
        if self.get('detection.threshold_method') not in ['otsu', 'max_entropy']:
            errors.append("detection.threshold_method must be 'otsu' or 'max_entropy'")
        
        # Validate merging parameters
        if self.get('merging.object_similarity_threshold') <= 0 or self.get('merging.object_similarity_threshold') > 1:
            errors.append("merging.object_similarity_threshold must be in (0, 1]")
        
        return errors
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("\n" + "="*80)
        print("Configuration Summary")
        print("="*80)
        
        sections = [
            ('Models', 'models'),
            ('Processing', 'processing'),
            ('Cost Weights', 'costs'),
            ('Detection', 'detection'),
            ('Merging', 'merging'),
            ('Cache', 'cache'),
            ('Visualization', 'visualization'),
            ('Output', 'output'),
        ]
        
        for section_name, section_key in sections:
            print(f"\n{section_name}:")
            self._print_dict(self.config.get(section_key, {}), indent=2)
        
        print("\n" + "="*80 + "\n")
    
    def _print_dict(self, d, indent=0):
        """Print dictionary with indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")
    
    def __getitem__(self, key):
        """Enable dictionary-style access."""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """Enable dictionary-style setting."""
        self.config[key] = value
