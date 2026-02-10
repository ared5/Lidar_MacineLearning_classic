"""
Pitch Angle Coleman Transformation Module

This module provides functions to transform pitch angles from rotating frame
to fixed frame using Coleman transformation. This is particularly useful for
turbines with Individual Pitch Control (IPC).

The transformation extracts:
- pitch_0: Collective pitch (average of all blades)
- pitch_1c, pitch_1s: 1P components (once-per-revolution, gravity effect)
- pitch_2c, pitch_2s: 2P components (twice-per-revolution)

Also calculates rates of change for all components.

Author: ML_Lidar_HubLoads Project
Date: 2026
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def create_pitch_coleman_transform(
    df: pd.DataFrame,
    pitch_blade1_col: str = 'Pitch angle 1',
    pitch_blade2_col: str = 'Pitch angle 2',
    azimuth_col: str = 'Rotor azimuth',
    include_rates: bool = True,
    include_2p: bool = False
) -> pd.DataFrame:
    """
    Apply Coleman transformation to pitch angles.
    
    Transforms pitch angles from rotating reference frame (individual blades)
    to fixed reference frame (0P, 1P, 2P components).
    
    For a 2-bladed turbine:
    - pitch_0 = (pitch_1 + pitch_2) / 2  (collective pitch)
    - pitch_delta = (pitch_1 - pitch_2) / 2  (differential pitch)
    - pitch_1c = pitch_delta * cos(azimuth)  (1P cosine component)
    - pitch_1s = pitch_delta * sin(azimuth)  (1P sine component)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with pitch angles and azimuth
    pitch_blade1_col : str, default='Pitch angle 1'
        Column name for blade 1 pitch angle (degrees)
    pitch_blade2_col : str, default='Pitch angle 2'
        Column name for blade 2 pitch angle (degrees)
    azimuth_col : str, default='Rotor azimuth'
        Column name for rotor azimuth angle (degrees)
    include_rates : bool, default=True
        Whether to calculate time derivatives (rates of change)
    include_2p : bool, default=False
        Whether to calculate 2P components (twice-per-revolution)
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns:
        - 'pitch_0': Collective pitch component
        - 'pitch_1c': 1P cosine component
        - 'pitch_1s': 1P sine component
        - 'pitch_2c': 2P cosine component (if include_2p=True)
        - 'pitch_2s': 2P sine component (if include_2p=True)
        - '*_rate' versions for all components (if include_rates=True)
    
    Notes
    -----
    - For turbines without IPC (Individual Pitch Control), pitch_1 == pitch_2,
      so pitch_delta ≈ 0, resulting in pitch_1c ≈ 0 and pitch_1s ≈ 0
    - For turbines with IPC, these components represent active control for
      load reduction
    
    Examples
    --------
    >>> df_with_pitch = create_pitch_coleman_transform(df)
    >>> # Creates pitch_0, pitch_1c, pitch_1s and their rates
    
    >>> df_with_2p = create_pitch_coleman_transform(df, include_2p=True)
    >>> # Also includes pitch_2c and pitch_2s
    """
    df = df.copy()
    
    # Verify required columns exist
    required_cols = [pitch_blade1_col, pitch_blade2_col, azimuth_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Extract data
    theta_1 = df[pitch_blade1_col].values
    theta_2 = df[pitch_blade2_col].values
    azimuth = df[azimuth_col].values
    
    # Convert azimuth to radians if in degrees
    if azimuth.max() > 7:  # Likely in degrees
        azimuth_rad = np.deg2rad(azimuth)
    else:
        azimuth_rad = azimuth
    
    # Calculate Coleman components
    # 0P: Collective (average)
    df['pitch_0'] = (theta_1 + theta_2) / 2.0
    
    # Differential component (for 1P)
    theta_delta = (theta_1 - theta_2) / 2.0
    
    # 1P: Once-per-revolution (gravity effect)
    df['pitch_1c'] = theta_delta * np.cos(azimuth_rad)
    df['pitch_1s'] = theta_delta * np.sin(azimuth_rad)
    
    # 2P: Twice-per-revolution (if requested)
    if include_2p:
        # For 2P, use sum instead of difference
        theta_sum = (theta_1 + theta_2) / 2.0
        df['pitch_2c'] = theta_sum * np.cos(2 * azimuth_rad)
        df['pitch_2s'] = theta_sum * np.sin(2 * azimuth_rad)
    
    # Calculate rates of change (time derivatives)
    if include_rates:
        df = calculate_pitch_rates(df, include_2p=include_2p)
    
    return df


def calculate_pitch_rates(
    df: pd.DataFrame,
    include_2p: bool = False,
    smooth_window: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate rates of change for pitch components.
    
    Computes time derivatives of pitch components using finite differences.
    Optionally applies smoothing to reduce noise.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with pitch components (pitch_0, pitch_1c, pitch_1s, etc.)
    include_2p : bool, default=False
        Whether to also calculate rates for 2P components
    smooth_window : int, optional
        Window size for moving average smoothing of rates.
        If None, no smoothing is applied
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added '*_rate' columns for each pitch component
    
    Notes
    -----
    Rates are calculated as:
    rate = d(pitch)/dt ≈ Δpitch / Δt
    
    If Time column exists, uses actual time differences.
    Otherwise, assumes constant sampling rate.
    """
    df = df.copy()
    
    # Define pitch components to differentiate
    pitch_components = ['pitch_0', 'pitch_1c', 'pitch_1s']
    if include_2p:
        pitch_components.extend(['pitch_2c', 'pitch_2s'])
    
    # Verify components exist
    available_components = [col for col in pitch_components if col in df.columns]
    
    if not available_components:
        raise ValueError("No pitch components found in dataframe")
    
    # Calculate time differences
    if 'Time' in df.columns:
        dt = df['Time'].diff()
        dt.fillna(df['Time'].iloc[1] - df['Time'].iloc[0], inplace=True)
    else:
        # Assume constant sampling (use 1 as normalized time step)
        dt = 1.0
    
    # Calculate rates for each component
    for component in available_components:
        rate_col = f'{component}_rate'
        df[rate_col] = df[component].diff() / dt
        
        # Fill first value (NaN from diff)
        df[rate_col].fillna(0, inplace=True)
        
        # Optional smoothing
        if smooth_window is not None and smooth_window > 1:
            df[rate_col] = df[rate_col].rolling(
                window=smooth_window,
                center=True
            ).mean()
            # Fill edges
            df[rate_col].fillna(method='bfill', inplace=True)
            df[rate_col].fillna(method='ffill', inplace=True)
    
    return df


def detect_ipc_presence(
    df: pd.DataFrame,
    pitch_blade1_col: str = 'Pitch angle 1',
    pitch_blade2_col: str = 'Pitch angle 2',
    threshold: float = 0.001
) -> Tuple[bool, dict]:
    """
    Detect if Individual Pitch Control (IPC) is present.
    
    Checks if pitch angles differ between blades, indicating active IPC.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with pitch angles
    pitch_blade1_col : str
        Column name for blade 1 pitch
    pitch_blade2_col : str
        Column name for blade 2 pitch
    threshold : float, default=0.001
        Maximum difference (degrees) to consider pitches as identical
    
    Returns
    -------
    has_ipc : bool
        True if IPC is detected (pitches differ significantly)
    stats : dict
        Statistics about pitch differences:
        - 'max_diff': Maximum absolute difference
        - 'mean_diff': Mean absolute difference
        - 'std_diff': Standard deviation of differences
    
    Examples
    --------
    >>> has_ipc, stats = detect_ipc_presence(df)
    >>> if has_ipc:
    ...     print("IPC detected! Using Coleman transform.")
    ... else:
    ...     print("No IPC. Pitch components will be near zero.")
    """
    theta_1 = df[pitch_blade1_col].values
    theta_2 = df[pitch_blade2_col].values
    
    # Calculate differences
    diff = np.abs(theta_1 - theta_2)
    
    # Statistics
    stats = {
        'max_diff': diff.max(),
        'mean_diff': diff.mean(),
        'std_diff': diff.std()
    }
    
    # Detection
    has_ipc = stats['max_diff'] > threshold
    
    return has_ipc, stats


def create_pitch_interaction_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create interaction features between pitch components.
    
    Generates additional features that capture relationships between
    different pitch components, which may be useful for ML models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with pitch components (must already have pitch_0, pitch_1c, pitch_1s)
    
    Returns
    -------
    pd.DataFrame
        Dataframe with added interaction features:
        - 'pitch_1_magnitude': Magnitude of 1P vector
        - 'pitch_1_phase': Phase angle of 1P vector
        - 'pitch_activity': Combined measure of pitch activity
    
    Notes
    -----
    Interaction features:
    - pitch_1_magnitude = sqrt(pitch_1c² + pitch_1s²)
    - pitch_1_phase = atan2(pitch_1s, pitch_1c)
    - pitch_activity = |pitch_0_rate| + pitch_1_magnitude
    """
    df = df.copy()
    
    # Verify required components exist
    required = ['pitch_0', 'pitch_1c', 'pitch_1s']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Required columns {required} not found")
    
    # 1P magnitude (vector norm)
    df['pitch_1_magnitude'] = np.sqrt(
        df['pitch_1c']**2 + df['pitch_1s']**2
    )
    
    # 1P phase angle
    df['pitch_1_phase'] = np.arctan2(df['pitch_1s'], df['pitch_1c'])
    
    # Pitch activity (if rates available)
    if 'pitch_0_rate' in df.columns:
        df['pitch_activity'] = (
            np.abs(df['pitch_0_rate']) + df['pitch_1_magnitude']
        )
    
    return df


def diagnose_pitch_coleman(
    df: pd.DataFrame,
    pitch_blade1_col: str = 'Pitch angle 1',
    pitch_blade2_col: str = 'Pitch angle 2',
    azimuth_col: str = 'Rotor azimuth'
) -> None:
    """
    Print diagnostic information about pitch Coleman transformation.
    
    Useful for debugging and understanding the data before applying
    the Coleman transformation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    pitch_blade1_col : str
        Blade 1 pitch column
    pitch_blade2_col : str
        Blade 2 pitch column
    azimuth_col : str
        Azimuth column
    
    Examples
    --------
    >>> diagnose_pitch_coleman(df)
    PITCH COLEMAN DIAGNOSTICS
    ========================
    Blade 1 pitch: min=5.2°, max=12.8°, mean=8.5°
    Blade 2 pitch: min=5.2°, max=12.8°, mean=8.5°
    Difference:    max=0.001°, mean=0.0001°
    IPC Status:    NOT DETECTED (collective pitch only)
    ...
    """
    print("="*70)
    print("PITCH COLEMAN DIAGNOSTICS")
    print("="*70)
    
    # Check if columns exist
    missing = []
    for col in [pitch_blade1_col, pitch_blade2_col, azimuth_col]:
        if col not in df.columns:
            missing.append(col)
    
    if missing:
        print(f"\n❌ Missing columns: {missing}")
        return
    
    # Extract data
    theta_1 = df[pitch_blade1_col].values
    theta_2 = df[pitch_blade2_col].values
    azimuth = df[azimuth_col].values
    
    # Statistics
    print(f"\n📊 PITCH STATISTICS:")
    print(f"   Blade 1: min={theta_1.min():.3f}°, max={theta_1.max():.3f}°, "
          f"mean={theta_1.mean():.3f}°, std={theta_1.std():.3f}°")
    print(f"   Blade 2: min={theta_2.min():.3f}°, max={theta_2.max():.3f}°, "
          f"mean={theta_2.mean():.3f}°, std={theta_2.std():.3f}°")
    
    # Difference analysis
    diff = np.abs(theta_1 - theta_2)
    print(f"\n   |Blade1 - Blade2|:")
    print(f"      min={diff.min():.3f}°, max={diff.max():.3f}°, "
          f"mean={diff.mean():.3f}°, std={diff.std():.3f}°")
    
    # IPC detection
    has_ipc, ipc_stats = detect_ipc_presence(
        df, pitch_blade1_col, pitch_blade2_col
    )
    
    print(f"\n🎯 IPC STATUS:")
    if has_ipc:
        print(f"   ✅ IPC DETECTED")
        print(f"      Pitches differ significantly (max diff = {ipc_stats['max_diff']:.3f}°)")
        print(f"      Coleman transform will produce meaningful 1P components")
    else:
        print(f"   ❌ NO IPC DETECTED")
        print(f"      Pitches are identical (max diff = {ipc_stats['max_diff']:.3f}°)")
        print(f"      This is NORMAL for collective pitch control")
        print(f"      Coleman components (pitch_1c, pitch_1s) will be ≈ 0")
    
    # Azimuth check
    print(f"\n📐 AZIMUTH:")
    if azimuth.max() > 7:
        print(f"   Units: Degrees (max={azimuth.max():.1f}°)")
    else:
        print(f"   Units: Radians (max={azimuth.max():.3f} rad)")
    
    print(f"   Range: {azimuth.min():.1f} to {azimuth.max():.1f}")
    
    # Preview transformed components
    print(f"\n📈 PREVIEW OF COLEMAN COMPONENTS:")
    theta_0 = (theta_1 + theta_2) / 2.0
    theta_delta = (theta_1 - theta_2) / 2.0
    
    azimuth_rad = np.deg2rad(azimuth) if azimuth.max() > 7 else azimuth
    theta_1c = theta_delta * np.cos(azimuth_rad)
    theta_1s = theta_delta * np.sin(azimuth_rad)
    
    print(f"   pitch_0 (collective): mean={theta_0.mean():.3f}°, std={theta_0.std():.3f}°")
    print(f"   pitch_1c (1P cos):    mean={theta_1c.mean():.3f}°, std={theta_1c.std():.3f}°")
    print(f"   pitch_1s (1P sin):    mean={theta_1s.mean():.3f}°, std={theta_1s.std():.3f}°")
    
    print("\n" + "="*70)
