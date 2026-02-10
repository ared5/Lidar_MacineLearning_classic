"""
Coleman Transform Module - Convert blade loads to fixed frame components.

This module implements the Coleman transformation to extract frequency components
(0P, 1P, 2P) from blade root moments in the rotating frame and project them
to the fixed frame.

Author: Wind ML Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from typing import Optional


def lowpass_filter(signal_data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """
    Apply a Butterworth lowpass filter to the signal.
    
    Args:
        signal_data: Input signal array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal array
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    # Ensure value is in valid range (0, 1)
    normal_cutoff = max(0.001, min(normal_cutoff, 0.999))
    
    sos = sp_signal.butter(order, normal_cutoff, btype='low', output='sos')
    filtered_signal = sp_signal.sosfilt(sos, signal_data)
    
    return filtered_signal


def bandpass_filter(signal_data: np.ndarray, lowcut: float, highcut: float, 
                    fs: float, order: int = 2) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the signal.
    
    Args:
        signal_data: Input signal array
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal array
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure values are in valid range (0, 1)
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))
    
    sos = sp_signal.butter(order, [low, high], btype='band', output='sos')
    filtered_signal = sp_signal.sosfilt(sos, signal_data)
    
    return filtered_signal


def bandpass_filter_safe(signal_data: np.ndarray, lowcut: float, highcut: float,
                         fs: float, order: int = 2) -> np.ndarray:
    """
    Safe version of bandpass filter that handles errors.
    
    Args:
        signal_data: Input signal array
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal, or original signal if filtering fails
    """
    try:
        return bandpass_filter(signal_data, lowcut, highcut, fs, order)
    except Exception as e:
        print(f"      WARNING: Bandpass filter failed ({e}). Using unfiltered signal.")
        return signal_data


def lowpass_filter_safe(signal_data: np.ndarray, cutoff: float, fs: float, 
                        order: int = 2) -> np.ndarray:
    """
    Safe version of lowpass filter that handles errors.
    
    Args:
        signal_data: Input signal array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal, or original signal if filtering fails
    """
    try:
        return lowpass_filter(signal_data, cutoff, fs, order)
    except Exception as e:
        print(f"      WARNING: Lowpass filter failed ({e}). Using unfiltered signal.")
        return signal_data


def create_frequency_components_1P_2P(df: pd.DataFrame, 
                                      apply_filtering: bool = True) -> pd.DataFrame:
    """
    Create 0P, 1P and 2P frequency components from blade root moments.
    
    DESCRIPTION:
    This function creates the following targets from blade moments M1(t) and M2(t):
    
    1. Sum and difference signals:
       - M_Σ(t) = (M1(t) + M2(t)) / 2  → contains even components (2P, 4P, ...)
       - M_Δ(t) = (M1(t) - M2(t)) / 2  → contains odd components (1P, 3P, ...)
    
    2. 0P component (slow/average):
       - M_0(t) = M_Σ(t)  → slow component
    
    3. 1P component (projected to fixed frame):
       - M_1c(t) = M_Δ(t) * cos(ψ(t))
       - M_1s(t) = M_Δ(t) * sin(ψ(t))
       where ψ(t) is the azimuth angle of blade 1
    
    4. 2P component (projected to fixed frame):
       - M_2c(t) = M_Σ(t) * cos(2ψ(t))
       - M_2s(t) = M_Σ(t) * sin(2ψ(t))
    
    IMPORTANT: For clean 1P and 2P, it's recommended to filter:
       - M_Δ around 1P before projection (band-pass)
       - M_Σ around 2P before projection (band-pass)
    
    Output targets: [M_0, M_1c, M_1s, M_2c, M_2s]
    
    Args:
        df: DataFrame with simulation data. Must contain:
            - 'Time': time in seconds
            - 'Rotor speed': rotor speed in rpm
            - 'Rotor azimuth angle': azimuth angle of rotor (blade 1)
            - 'Blade root 1 My': blade 1 bending moment
            - 'Blade root 2 My': blade 2 bending moment
        apply_filtering: If True, apply band-pass filtering before projection
        
    Returns:
        DataFrame with new columns:
            - 'M_0' (0P): slow component
            - 'M_1c' (1P cosine): 1P in-phase component
            - 'M_1s' (1P sine): 1P quadrature component
            - 'M_2c' (2P cosine): 2P in-phase component
            - 'M_2s' (2P sine): 2P quadrature component
    
    Raises:
        ValueError: If required columns are missing in DataFrame
    """
    
    # Validate required columns
    required_cols = ['Time', 'Rotor speed', 'Rotor azimuth angle', 
                     'Blade root 1 My', 'Blade root 2 My']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    print("=" * 70)
    print("Creating 0P, 1P and 2P frequency components...")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Get basic parameters
    # =========================================================================
    M1 = df['Blade root 1 My'].values
    M2 = df['Blade root 2 My'].values
    time = df['Time'].values
    azimuth = df['Rotor azimuth angle'].values
    rotor_speed_rpm = df['Rotor speed'].values
    
    # Convert azimuth to radians if in degrees
    if azimuth.max() > 6.5:
        azimuth_rad = np.deg2rad(azimuth)
        print("   Azimuth converted from degrees to radians")
    else:
        azimuth_rad = azimuth
        print("   Azimuth already in radians")
    
    # Calculate frequencies
    freq_1P_Hz = rotor_speed_rpm / (2 * np.pi)  # Convert rad/s to Hz
    freq_2P_Hz = 2 * freq_1P_Hz
    freq_1P_mean = freq_1P_Hz.mean()
    freq_2P_mean = freq_2P_Hz.mean()
    
    # Calculate sampling frequency
    if len(df) > 1:
        dt = time[1] - time[0]
        fs = 1.0 / dt
    else:
        dt = 0.02
        fs = 50.0
    
    print(f"\n   Parameters:")
    print(f"   - Average Rotor Speed: {rotor_speed_rpm.mean():.2f} rpm")
    print(f"   - Average 1P frequency: {freq_1P_mean:.3f} Hz")
    print(f"   - Average 2P frequency: {freq_2P_mean:.3f} Hz")
    print(f"   - Sampling frequency: {fs:.1f} Hz")
    print(f"   - Number of samples: {len(df)}")
    
    # =========================================================================
    # STEP 2: Calculate sum (Σ) and difference (Δ) signals
    # =========================================================================
    print(f"\n   Calculating M_Σ and M_Δ...")
    
    M_sum = (M1 + M2) / 2.0  # M_Σ: contains even components (2P, 4P, ...)
    M_diff = (M1 - M2) / 2.0  # M_Δ: contains odd components (1P, 3P, ...)
    
    print(f"   - M_Σ (sum) calculated: range [{M_sum.min():.2f}, {M_sum.max():.2f}]")
    print(f"   - M_Δ (difference) calculated: range [{M_diff.min():.2f}, {M_diff.max():.2f}]")
    
    # =========================================================================
    # STEP 3: Apply band-pass filtering (optional but recommended)
    # =========================================================================
    if apply_filtering:
        print(f"\n   Applying band-pass filtering...")
        
        # Filter M_Δ around 1P
        bandwidth_1P = 0.3  # Bandwidth in Hz around 1P
        lowcut_1P = max(0.01, freq_1P_mean - bandwidth_1P)
        highcut_1P = min(fs/2 - 0.1, freq_1P_mean + bandwidth_1P)
        
        print(f"   - Filtering M_Δ around 1P: [{lowcut_1P:.3f}, {highcut_1P:.3f}] Hz")
        M_diff_filtered = bandpass_filter_safe(M_diff, lowcut_1P, highcut_1P, fs, order=2)
        
        # Filter M_Σ around 2P
        bandwidth_2P = 0.5  # Bandwidth in Hz around 2P
        lowcut_2P = max(0.01, freq_2P_mean - bandwidth_2P)
        highcut_2P = min(fs/2 - 0.1, freq_2P_mean + bandwidth_2P)
        
        print(f"   - Filtering M_Σ around 2P: [{lowcut_2P:.3f}, {highcut_2P:.3f}] Hz")
        M_sum_filtered = bandpass_filter_safe(M_sum, lowcut_2P, highcut_2P, fs, order=2)
    else:
        print(f"\n   No filtering (apply_filtering=False)")
        M_diff_filtered = M_diff
        M_sum_filtered = M_sum
    
    # =========================================================================
    # STEP 4: Create 0P, 1P and 2P components
    # =========================================================================
    print(f"\n   Creating frequency components...")
    
    # 0P: DC component (remove even frequencies 2P, 4P, ...)
    if apply_filtering:
        # Low-pass filter to keep only DC component
        # Cut below 1P to eliminate 2P, 4P, etc.
        cutoff_0P = freq_1P_mean * 5  # Cut at half of 1P
        print(f"   - Filtering M_0 (low-pass) with cutoff at {cutoff_0P:.3f} Hz")
        M_0 = lowpass_filter_safe(M_sum, cutoff_0P, fs, order=2)
        print(f"   - M_0 (0P): DC component created (without 2P, 4P, ...)")
    else:
        M_0 = M_sum  # Unfiltered
        print(f"   - M_0 (0P): slow component created (unfiltered)")
    
    # 1P: Projection of M_Δ to fixed frame using azimuth
    M_1c = M_diff_filtered * np.cos(azimuth_rad)  # 1P in-phase component (cosine)
    M_1s = M_diff_filtered * np.sin(azimuth_rad)  # 1P quadrature component (sine)
    print(f"   - M_1c, M_1s (1P): components created with fixed-frame projection")
    
    # 2P: Projection of M_Σ to fixed frame using 2*azimuth
    M_2c = M_sum_filtered * np.cos(2 * azimuth_rad)  # 2P in-phase component (cosine)
    M_2s = M_sum_filtered * np.sin(2 * azimuth_rad)  # 2P quadrature component (sine)
    print(f"   - M_2c, M_2s (2P): components created with fixed-frame projection")
    
    # =========================================================================
    # STEP 5: Add to DataFrame
    # =========================================================================
    print(f"\n   Adding columns to DataFrame...")
    
    df['M_0'] = M_0      # 0P
    df['M_1c'] = M_1c    # 1P cosine
    df['M_1s'] = M_1s    # 1P sine
    df['M_2c'] = M_2c    # 2P cosine
    df['M_2s'] = M_2s    # 2P sine
    
    new_columns = ['M_0', 'M_1c', 'M_1s', 'M_2c', 'M_2s']
    print(f"   - Columns created: {new_columns}")
    
    # =========================================================================
    # STEP 6: Final summary
    # =========================================================================
    print(f"\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"=" * 70)
    print(f"   Output vector: y(t) = [M_0, M_1c, M_1s, M_2c, M_2s]")
    print(f"   - M_0:  0P component (slow)")
    print(f"   - M_1c: 1P in-phase component (cosine)")
    print(f"   - M_1s: 1P quadrature component (sine)")
    print(f"   - M_2c: 2P in-phase component (cosine)")
    print(f"   - M_2s: 2P quadrature component (sine)")
    print(f"\n   Final DataFrame shape: {df.shape}")
    print(f"=" * 70)
    
    return df
