"""
Settings Module - Central configuration management for Wind Turbine ML project.

This module provides utilities to load and access YAML configuration files
and manage project paths.

Author: Wind ML Team
Date: February 2026
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Configuration manager for the project.
    Loads YAML files and provides easy access to configuration parameters.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            project_root: Root directory of the project. If None, auto-detect.
        """
        if project_root is None:
            # Auto-detect project root (assumes this file is in src/windml/config/)
            self.project_root = Path(__file__).parent.parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.configs_dir = self.project_root / 'configs'
        
        # Load configurations
        self.paths = self._load_config('paths.yaml')
        self.features = self._load_config('features.yaml')
        self.models = self._load_config('models.yaml')
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            filename: Name of the config file in configs/ directory
            
        Returns:
            Dictionary with configuration parameters
        """
        config_path = self.configs_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_path(self, *keys: str) -> Path:
        """
        Get a path from paths.yaml configuration.
        
        Args:
            *keys: Nested keys to access the path (e.g., 'data', 'processed')
            
        Returns:
            Absolute Path object
            
        Example:
            >>> config = Config()
            >>> config.get_path('data', 'processed')
            PosixPath('/path/to/project/data/processed')
        """
        value = self.paths
        for key in keys:
            value = value[key]
        
        # Convert to absolute path
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        
        return path
    
    def get_feature_config(self, *keys: str) -> Any:
        """
        Get a configuration value from features.yaml.
        
        Args:
            *keys: Nested keys to access the value
            
        Returns:
            Configuration value (can be dict, list, str, etc.)
        """
        value = self.features
        for key in keys:
            value = value[key]
        return value
    
    def get_model_config(self, *keys: str) -> Any:
        """
        Get a configuration value from models.yaml.
        
        Args:
            *keys: Nested keys to access the value
            
        Returns:
            Configuration value (can be dict, list, str, etc.)
        """
        value = self.models
        for key in keys:
            value = value[key]
        return value
    
    def ensure_directories(self):
        """
        Create all necessary directories defined in paths.yaml if they don't exist.
        """
        # Create data directories
        for key in ['raw', 'interim', 'processed', 'ml_traditional']:
            dir_path = self.get_path('data', key)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create model directories
        for key in ['root', 'random_forest', 'xgboost', 'xgboost_nonorm', 
                    'xgboost_individual', 'scalers']:
            dir_path = self.get_path('models', key)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create report directories
        for key in ['root', 'figures', 'tables', 'eda']:
            dir_path = self.get_path('reports', key)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✅ All project directories created successfully")


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(project_root: Optional[Path] = None) -> Config:
    """
    Get or create the global configuration instance (singleton pattern).
    
    Args:
        project_root: Root directory of the project. If None, auto-detect.
        
    Returns:
        Config instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config(project_root)
    
    return _global_config


# Convenience functions for quick access
def get_path(*keys: str) -> Path:
    """Get a path from configuration."""
    return get_config().get_path(*keys)


def get_feature_config(*keys: str) -> Any:
    """Get feature configuration."""
    return get_config().get_feature_config(*keys)


def get_model_config(*keys: str) -> Any:
    """Get model configuration."""
    return get_config().get_model_config(*keys)


# Constants for convenience
RANDOM_STATE = 42
DT_DEFAULT = 0.02  # Default sampling time (seconds) - 50 Hz
