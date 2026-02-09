"""
Signal Processing Module - Functions for creating lagged features and transformations.

This module provides functions to create time-lagged features for VLOS (Line-of-Sight
velocity) signals and other time series transformations.

Author: Wind ML Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def create_vlos_lags(
    df: pd.DataFrame,
    lag_seconds_list: List[int] = [2, 5, 8, 11, 14, 17, 20, 23, 26],
    range_values: Optional[List[int]] = None,
    range_min: Optional[int] = None,
    range_max: Optional[int] = None,
    include_all_ranges: bool = False,
) -> pd.DataFrame:
    """
    Create lagged features for wind velocity (VLOS) variables.
    
    This function creates time-shifted versions of VLOS measurements to capture
    temporal dynamics. Useful for ML models that need historical wind information
    to predict blade loads.
    
    Args:
        df: DataFrame with the data
        lag_seconds_list: List of lags in seconds to create
        range_values: List/tuple of specific ranges (e.g. [5, 7, 9])
        range_min: Minimum range (inclusive) if using range filter
        range_max: Maximum range (inclusive) if using range filter
        include_all_ranges: If True, ignore filters and use all VLOS columns
        
    Returns:
        DataFrame with new lagged columns added
        
    Example:
        >>> df = create_vlos_lags(df, lag_seconds_list=[5, 10, 15], range_values=[5])
        >>> # Creates: LAC_VLOS_BEAM0_RANGE5_lag5s, LAC_VLOS_BEAM0_RANGE5_lag10s, etc.
    """
    # Identify VLOS columns
    vlos_columns = [col for col in df.columns if col.startswith('LAC_VLOS')]
    
    # Filter by range if requested
    if not include_all_ranges:
        filtered_columns = []
        for col in vlos_columns:
            parts = col.split('_')
            range_part = next((p for p in parts if p.startswith('RANGE')), None)
            if not range_part:
                continue
            try:
                range_value = int(range_part.replace('RANGE', ''))
            except ValueError:
                continue
            
            if range_values is not None:
                if range_value in range_values:
                    filtered_columns.append(col)
            elif range_min is not None or range_max is not None:
                min_ok = range_min is None or range_value >= range_min
                max_ok = range_max is None or range_value <= range_max
                if min_ok and max_ok:
                    filtered_columns.append(col)
        
        vlos_columns = filtered_columns
    
    print(f"VLOS variables found: {len(vlos_columns)}")
    for col in vlos_columns:
        print(f"  - {col}")
    
    # Calculate sampling time (dt) assuming Time column exists
    if 'Time' in df.columns and len(df) > 1:
        dt = df['Time'].iloc[1] - df['Time'].iloc[0]  # Seconds between samples
        print(f"\nDetected sampling time: {dt:.4f} seconds")
    else:
        dt = 0.02  # Default 50Hz
        print(f"\nDefault sampling time: {dt} seconds")
    
    # Create lags for each VLOS variable
    print(f"\nCreating {len(lag_seconds_list)} lags for each VLOS variable...")
    
    total_created = 0
    for vlos_col in vlos_columns:
        for lag_sec in lag_seconds_list:
            # Calculate number of samples for the lag
            lag_samples = int(round(lag_sec / dt))
            
            # Create name for new column
            new_col_name = f"{vlos_col}_lag{lag_sec}s"
            
            # Create column with shift
            df[new_col_name] = df[vlos_col].shift(lag_samples)
            
            total_created += 1
    
    print(f"Total lag features created: {total_created}")
    print(f"DataFrame shape: {df.shape}")
    
    return df


def create_trigonometric_components(
    df: pd.DataFrame,
    angle_columns: List[str],
    output_prefixes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create sine and cosine components from angular variables.
    
    This avoids discontinuities at 0-360 degrees and helps ML models learn
    cyclic patterns better.
    
    Args:
        df: DataFrame with the data
        angle_columns: List of column names with angular data
        output_prefixes: Optional custom prefixes for output columns.
                        If None, uses "sin_{col}" and "cos_{col}"
        
    Returns:
        DataFrame with new sin/cos columns added
        
    Example:
        >>> df = create_trigonometric_components(df, ['Rotor azimuth angle'])
        >>> # Creates: sin_rotor_azimuth, cos_rotor_azimuth
    """
    print("=" * 70)
    print("CREATING TRIGONOMETRIC COMPONENTS")
    print("=" * 70)
    
    for idx, angle_col in enumerate(angle_columns):
        if angle_col not in df.columns:
            print(f"WARNING: Column '{angle_col}' not found - skipping")
            continue
        
        print(f"\nProcessing '{angle_col}'...")
        
        # Check range to determine units
        max_val = df[angle_col].abs().max()
        min_val = df[angle_col].min()
        
        # If value > 2π (6.28), probably in degrees
        if max_val > 6.5:
            print(f"   Range detected: [{min_val:.1f}, {max_val:.1f}] (degrees)")
            angle_rad = np.deg2rad(df[angle_col])
        else:
            print(f"   Range detected: [{min_val:.3f}, {max_val:.3f}] (radians)")
            angle_rad = df[angle_col]
        
        # Determine output names
        if output_prefixes and idx < len(output_prefixes):
            sin_name = f"sin_{output_prefixes[idx]}"
            cos_name = f"cos_{output_prefixes[idx]}"
        else:
            # Clean name: remove spaces and special characters
            clean_name = angle_col.lower().replace(' ', '_').replace('angle', '').strip('_')
            sin_name = f"sin_{clean_name}"
            cos_name = f"cos_{clean_name}"
        
        # Create components
        df[sin_name] = np.sin(angle_rad)
        df[cos_name] = np.cos(angle_rad)
        
        print(f"   ✓ Created: {sin_name}, {cos_name}")
    
    print(f"\n{'='*70}")
    print(f"Final DataFrame shape: {df.shape}")
    print("="*70)
    
    return df


def create_azimuth_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sine and cosine components of rotor azimuth angle.
    
    This is a convenience function specifically for azimuth angle, which is
    commonly used in wind turbine analysis.
    
    Args:
        df: DataFrame with the data
        
    Returns:
        DataFrame with sin_rotor_azimuth and cos_rotor_azimuth columns
    """
    azimuth_col = 'Rotor azimuth angle'
    
    if azimuth_col not in df.columns:
        print(f"WARNING: Column '{azimuth_col}' not found")
        return df
    
    print(f"Creating trigonometric components of '{azimuth_col}'...")
    
    # Check range to determine units
    max_val = df[azimuth_col].max()
    
    if max_val > 6.5:  # If > 2*pi, probably in degrees
        print(f"   Range detected: 0-{max_val:.1f} (degrees)")
        azimuth_rad = np.deg2rad(df[azimuth_col])
    else:
        print(f"   Range detected: 0-{max_val:.1f} (radians)")
        azimuth_rad = df[azimuth_col]
    
    # Create components
    df['sin_rotor_azimuth'] = np.sin(azimuth_rad)
    df['cos_rotor_azimuth'] = np.cos(azimuth_rad)
    
    print(f"   ✓ Created 2 new columns: sin_rotor_azimuth, cos_rotor_azimuth")
    print(f"   Shape: {df.shape}")
    
    return df


def create_yawerror_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sine and cosine components of yaw error angle.
    
    Yaw error is the misalignment angle between the nacelle and the wind direction.
    Converting to sin/cos helps ML models learn better cyclic patterns.
    
    Args:
        df: DataFrame with the data
        
    Returns:
        DataFrame with sin_yawerror and cos_yawerror columns
    """
    yawerror_col = 'OPER_MEAS_YAWERROR'
    
    if yawerror_col not in df.columns:
        print(f"WARNING: Column '{yawerror_col}' not found")
        return df
    
    print(f"Creating trigonometric components of '{yawerror_col}'...")
    
    # Check range to determine units
    max_val = df[yawerror_col].abs().max()
    min_val = df[yawerror_col].min()
    
    # Yaw error typically in radians range [-π, π] or degrees [-180°, 180°]
    if max_val > 6.5:  # If > 2*pi, probably in degrees
        print(f"   Range detected: [{min_val:.1f}, {max_val:.1f}] (degrees)")
        yawerror_rad = np.deg2rad(df[yawerror_col])
    else:
        print(f"   Range detected: [{min_val:.3f}, {max_val:.3f}] (radians)")
        yawerror_rad = df[yawerror_col]
    
    # Create trigonometric components
    df['sin_yawerror'] = np.sin(yawerror_rad)
    df['cos_yawerror'] = np.cos(yawerror_rad)
    
    print(f"   ✓ Created 2 new columns: sin_yawerror, cos_yawerror")
    print(f"   💡 Benefit: Model can learn better cyclic yaw error patterns")
    print(f"   Shape: {df.shape}")
    
    return df


def create_rolling_statistics(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    statistics: List[str] = ['mean', 'std']
) -> pd.DataFrame:
    """
    Create rolling window statistics for specified columns.
    
    Args:
        df: DataFrame with the data
        columns: List of column names to compute statistics on
        windows: List of window sizes (in number of samples)
        statistics: List of statistics to compute ('mean', 'std', 'min', 'max')
        
    Returns:
        DataFrame with new rolling statistics columns
    """
    print("=" * 70)
    print("CREATING ROLLING STATISTICS")
    print("=" * 70)
    
    total_created = 0
    
    for col in columns:
        if col not in df.columns:
            print(f"WARNING: Column '{col}' not found - skipping")
            continue
        
        for window in windows:
            for stat in statistics:
                new_col_name = f"{col}_rolling{window}_{stat}"
                
                if stat == 'mean':
                    df[new_col_name] = df[col].rolling(window=window, min_periods=1).mean()
                elif stat == 'std':
                    df[new_col_name] = df[col].rolling(window=window, min_periods=1).std()
                elif stat == 'min':
                    df[new_col_name] = df[col].rolling(window=window, min_periods=1).min()
                elif stat == 'max':
                    df[new_col_name] = df[col].rolling(window=window, min_periods=1).max()
                else:
                    print(f"WARNING: Unknown statistic '{stat}' - skipping")
                    continue
                
                total_created += 1
    
    print(f"\n Total rolling statistics created: {total_created}")
    print(f" Final DataFrame shape: {df.shape}")
    print("=" * 70)
    
    return df
