"""
Wind Field Statistics Module

This module provides functions to calculate wind field statistics including:
- Mean wind speed (U_mean)
- Wind speed standard deviation (U_std)
- Wind shear (vertical wind gradient)
- Rolling statistics for wind parameters

These features are commonly used in wind turbine load prediction.

Author: ML_Lidar_HubLoads Project
Date: 2026
"""

import numpy as np
import pandas as pd
from typing import Optional, List


def calculate_wind_statistics(
    df: pd.DataFrame,
    wind_speed_col: str = 'Hub wind speed Ux',
    window_size: Optional[int] = None,
    include_shear: bool = False
) -> pd.DataFrame:
    """
    Calculate wind field statistics.
    
    Creates features for mean wind speed and standard deviation. Optionally
    calculates wind shear if multiple height measurements are available.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with wind speed measurements
    wind_speed_col : str, default='Hub wind speed Ux'
        Name of the wind speed column to use
    window_size : int, optional
        Rolling window size in samples for temporal statistics.
        If None, calculates statistics for entire dataset
    include_shear : bool, default=False
        Whether to calculate wind shear (requires multiple height measurements)
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns:
        - 'U_mean': Mean wind speed
        - 'U_std': Wind speed standard deviation
        - 'U_shear': Wind shear (if include_shear=True)
    
    Examples
    --------
    >>> df_with_stats = calculate_wind_statistics(df, window_size=100)
    >>> # Creates U_mean and U_std with 100-sample rolling window
    
    >>> df_with_shear = calculate_wind_statistics(df, include_shear=True)
    >>> # Also calculates wind shear from multi-height measurements
    """
    df = df.copy()
    
    # Verify wind speed column exists
    if wind_speed_col not in df.columns:
        raise ValueError(f"Column '{wind_speed_col}' not found in dataframe")
    
    # Calculate statistics
    if window_size is not None:
        # Rolling window statistics
        df['U_mean'] = df[wind_speed_col].rolling(
            window=window_size, 
            center=True
        ).mean()
        df['U_std'] = df[wind_speed_col].rolling(
            window=window_size, 
            center=True
        ).std()
        
        # Fill NaN at edges with first/last valid values
        df['U_mean'].fillna(method='bfill', inplace=True)
        df['U_mean'].fillna(method='ffill', inplace=True)
        df['U_std'].fillna(value=0, inplace=True)
        
    else:
        # Global statistics (constant for entire dataset)
        df['U_mean'] = df[wind_speed_col].mean()
        df['U_std'] = df[wind_speed_col].std()
    
    # Calculate wind shear if requested
    if include_shear:
        df = calculate_wind_shear(df)
    
    return df


def calculate_wind_shear(
    df: pd.DataFrame,
    lower_speed_col: str = 'Hub wind speed Ux',
    upper_speed_col: Optional[str] = None,
    height_diff: float = 10.0
) -> pd.DataFrame:
    """
    Calculate wind shear from multi-height wind speed measurements.
    
    Wind shear represents the vertical gradient of wind speed, which affects
    turbine loads. Can be calculated from direct measurements at different
    heights or estimated from turbulence parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    lower_speed_col : str, default='Hub wind speed Ux'
        Column name for lower height wind speed
    upper_speed_col : str, optional
        Column name for upper height wind speed.
        If None, estimates shear from turbulence
    height_diff : float, default=10.0
        Vertical distance between measurements (meters)
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added 'U_shear' column (m/s per meter)
    
    Notes
    -----
    If direct measurements at multiple heights are not available, the function
    can estimate shear from standard atmospheric boundary layer assumptions.
    """
    df = df.copy()
    
    if upper_speed_col is not None and upper_speed_col in df.columns:
        # Calculate from direct measurements
        df['U_shear'] = (df[upper_speed_col] - df[lower_speed_col]) / height_diff
    else:
        # Estimate using power law with typical exponent
        # Shear ~ U * alpha / height, where alpha is shear exponent (typically 0.1-0.2)
        alpha = 0.15  # Typical value for neutral atmosphere
        hub_height = 90.0  # Typical hub height in meters
        
        if 'U_mean' in df.columns:
            df['U_shear'] = df['U_mean'] * alpha / hub_height
        else:
            df['U_shear'] = df[lower_speed_col] * alpha / hub_height
    
    return df


def create_wind_direction_features(
    df: pd.DataFrame,
    wind_direction_col: str = 'Hub wind direction',
    yaw_col: str = 'Stationary hub Mx'
) -> pd.DataFrame:
    """
    Create features from wind direction and yaw angle.
    
    Calculates:
    - Yaw error (misalignment between turbine and wind)
    - Trigonometric components of wind direction
    - Rate of change of yaw error
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    wind_direction_col : str
        Column name for wind direction (degrees)
    yaw_col : str
        Column name for turbine yaw angle (degrees)
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added columns:
        - 'yaw_error': Difference between wind direction and yaw (degrees)
        - 'yaw_error_cos': Cosine component of yaw error
        - 'yaw_error_sin': Sine component of yaw error
        - 'yaw_error_rate': Rate of change of yaw error
    """
    df = df.copy()
    
    if wind_direction_col in df.columns and yaw_col in df.columns:
        # Calculate yaw error
        df['yaw_error'] = df[wind_direction_col] - df[yaw_col]
        
        # Normalize to [-180, 180] range
        df['yaw_error'] = ((df['yaw_error'] + 180) % 360) - 180
        
        # Trigonometric components
        yaw_error_rad = np.deg2rad(df['yaw_error'])
        df['yaw_error_cos'] = np.cos(yaw_error_rad)
        df['yaw_error_sin'] = np.sin(yaw_error_rad)
        
        # Rate of change
        time_col = 'Time' if 'Time' in df.columns else None
        if time_col:
            dt = df[time_col].diff()
            df['yaw_error_rate'] = df['yaw_error'].diff() / dt
            df['yaw_error_rate'].fillna(0, inplace=True)
        else:
            df['yaw_error_rate'] = df['yaw_error'].diff()
            df['yaw_error_rate'].fillna(0, inplace=True)
    
    return df


def create_turbulence_intensity(
    df: pd.DataFrame,
    wind_speed_col: str = 'Hub wind speed Ux',
    window_size: int = 100
) -> pd.DataFrame:
    """
    Calculate turbulence intensity (TI = std/mean).
    
    Turbulence intensity is a key parameter in wind energy that represents
    the level of turbulence normalized by mean wind speed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    wind_speed_col : str
        Column name for wind speed
    window_size : int, default=100
        Rolling window size for TI calculation
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added 'TI' column (turbulence intensity)
    
    Notes
    -----
    TI = sigma_u / U_mean, where:
    - sigma_u is the standard deviation of wind speed
    - U_mean is the mean wind speed
    
    Typical values range from 0.1 (low turbulence) to 0.3 (high turbulence).
    """
    df = df.copy()
    
    if wind_speed_col not in df.columns:
        raise ValueError(f"Column '{wind_speed_col}' not found")
    
    # Calculate rolling statistics
    U_mean = df[wind_speed_col].rolling(window=window_size, center=True).mean()
    U_std = df[wind_speed_col].rolling(window=window_size, center=True).std()
    
    # Calculate TI (avoid division by zero)
    df['TI'] = np.where(U_mean > 0.1, U_std / U_mean, 0)
    
    # Fill NaN values
    df['TI'].fillna(method='bfill', inplace=True)
    df['TI'].fillna(method='ffill', inplace=True)
    df['TI'].fillna(value=0.1, inplace=True)  # Default TI if all else fails
    
    return df


def create_all_windfield_features(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Create all wind field features in one function.
    
    This is a convenience function that applies all wind field feature
    engineering transformations based on configuration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    config : dict, optional
        Configuration dictionary specifying which features to create.
        If None, creates all available features.
        
        Expected keys:
        - 'wind_statistics': bool, whether to calculate U_mean, U_std
        - 'wind_shear': bool, whether to calculate shear
        - 'wind_direction': bool, whether to create direction features
        - 'turbulence_intensity': bool, whether to calculate TI
        - 'window_size': int, rolling window size
    
    Returns
    -------
    pd.DataFrame
        Dataframe with all requested wind field features
    
    Examples
    --------
    >>> config = {
    ...     'wind_statistics': True,
    ...     'wind_shear': True,
    ...     'window_size': 100
    ... }
    >>> df_with_features = create_all_windfield_features(df, config)
    """
    if config is None:
        # Default configuration: create all features
        config = {
            'wind_statistics': True,
            'wind_shear': False,
            'wind_direction': False,
            'turbulence_intensity': True,
            'window_size': 100
        }
    
    df = df.copy()
    
    # Wind statistics
    if config.get('wind_statistics', True):
        df = calculate_wind_statistics(
            df,
            window_size=config.get('window_size', 100),
            include_shear=config.get('wind_shear', False)
        )
    
    # Wind direction features
    if config.get('wind_direction', False):
        df = create_wind_direction_features(df)
    
    # Turbulence intensity
    if config.get('turbulence_intensity', True):
        df = create_turbulence_intensity(
            df,
            window_size=config.get('window_size', 100)
        )
    
    return df
