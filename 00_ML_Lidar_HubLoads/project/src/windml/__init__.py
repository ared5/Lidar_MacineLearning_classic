"""
Wind ML Package - Machine Learning for Wind Turbine Load Prediction.

This package contains modules for:
- Data loading and processing from Bladed simulations
- Feature engineering (Coleman transform, lags, wind statistics, pitch angles)
- Data assembly and time series splitting
- Normalization with independent scalers
- Model training (XGBoost, Random Forest, Ridge)
- Evaluation and validation
- Visualization and reporting

Author: Wind ML Team
Date: February 2026
"""

__version__ = "1.0.0"
__author__ = "Wind ML Team"

# Configuration
from .config.settings import get_config, get_path, get_feature_config, get_model_config

# Data loading and assembly
from .data.bladed_io import create_timeseries_csv, load_bladed_data
from .data.assemble import assemble_csvs, load_or_assemble_dataset, optimize_dataframe_dtypes
from .data.split import (
    identify_time_series,
    train_test_split_series,
    create_series_mapping,
    print_split_summary
)

# Feature engineering - Coleman transform (loads)
from .features.coleman import create_frequency_components_1P_2P, bandpass_filter, lowpass_filter

# Feature engineering - Signal processing (VLOS, angles)
from .features.signal import (
    create_vlos_lags,
    create_azimuth_components,
    create_yawerror_components,
    create_trigonometric_components
)

# Feature engineering - Wind field statistics  
from .features.windfield import (
    calculate_wind_statistics,
    calculate_wind_shear,
    create_turbulence_intensity,
    create_all_windfield_features
)

# Feature engineering - Pitch Coleman transform
from .features.angles import (
    create_pitch_coleman_transform,
    detect_ipc_presence,
    diagnose_pitch_coleman
)

# Preprocessing - Normalization
from .preprocessing.normalize import (
    create_independent_scalers,
    normalize_with_independent_scalers,
    denormalize_with_independent_scalers,
    fit_and_transform_independent,
    save_scalers,
    load_scalers
)

# Model training - XGBoost
from .modeling.train_xgb import (
    train_xgboost_multioutput,
    train_xgboost_individual,
    predict_xgboost
)

# Model training - Random Forest
from .modeling.train_rf import (
    train_random_forest,
    evaluate_random_forest,
    get_feature_importance,
    save_random_forest,
    load_random_forest
)

__all__ = [
    # Configuration
    'get_config',
    'get_path',
    'get_feature_config',
    'get_model_config',
    
    # Data loading and assembly
    'create_timeseries_csv',
    'load_bladed_data',
    'assemble_csvs',
    'load_or_assemble_dataset',
    'optimize_dataframe_dtypes',
    
    # Data splitting
    'identify_time_series',
    'train_test_split_series',
    'create_series_mapping',
    'print_split_summary',
    
    # Feature engineering - Coleman (loads)
    'create_frequency_components_1P_2P',
    'bandpass_filter',
    'lowpass_filter',
    
    # Feature engineering - Signals
    'create_vlos_lags',
    'create_azimuth_components',
    'create_yawerror_components',
    'create_trigonometric_components',
    
    # Feature engineering - Wind field
    'calculate_wind_statistics',
    'calculate_wind_shear',
    'create_turbulence_intensity',
    'create_all_windfield_features',
    
    # Feature engineering - Pitch angles
    'create_pitch_coleman_transform',
    'detect_ipc_presence',
    'diagnose_pitch_coleman',
    
    # Preprocessing - Normalization
    'create_independent_scalers',
    'normalize_with_independent_scalers',
    'denormalize_with_independent_scalers',
    'fit_and_transform_independent',
    'save_scalers',
    'load_scalers',
    
    # Model training - XGBoost
    'train_xgboost_multioutput',
    'train_xgboost_individual',
    'predict_xgboost',
    
    # Model training - Random Forest
    'train_random_forest',
    'evaluate_random_forest',
    'get_feature_importance',
    'save_random_forest',
    'load_random_forest',
]
