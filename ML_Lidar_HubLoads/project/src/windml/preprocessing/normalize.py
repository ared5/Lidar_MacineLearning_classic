"""
Normalization and Scaling Module

Functions for normalizing/scaling features and targets with independent scalers
per variable. This approach allows proper denormalization and handles different
variable scales appropriately.

Author: ML_Lidar_HubLoads Project
Date: 2026
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Tuple, Optional, Union
import joblib
from pathlib import Path


def create_independent_scalers(
    df: pd.DataFrame,
    scaler_type: str = 'standard'
) -> Dict[str, StandardScaler]:
    """
    Create independent scaler for each column.
    
    Instead of a single scaler for all features, creates one scaler per
    column. This allows:
    - Proper denormalization of individual columns
    - Different scaling strategies per variable if needed
    - Better handling of variables with different scales
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to create scalers for
    scaler_type : str, default='standard'
        Type of scaler to create:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'minmax': MinMaxScaler (range 0-1)
        - 'robust': RobustScaler (median, IQR-based)
    
    Returns
    -------
    dict
        Dictionary mapping column names to fitted scaler objects
    
    Examples
    --------
    >>> scalers_X = create_independent_scalers(X_train, scaler_type='standard')
    >>> # Each column has its own scaler
    >>> print(len(scalers_X))  # == number of columns in X_train
    """
    scalers = {}
    
    # Choose scaler class
    if scaler_type == 'standard':
        ScalerClass = StandardScaler
    elif scaler_type == 'minmax':
        ScalerClass = MinMaxScaler
    elif scaler_type == 'robust':
        ScalerClass = RobustScaler
    else:
        raise ValueError(
            f"Unknown scaler_type '{scaler_type}'. "
            f"Use 'standard', 'minmax', or 'robust'"
        )
    
    # Create and fit one scaler per column
    for col in df.columns:
        scaler = ScalerClass()
        scaler.fit(df[[col]])
        scalers[col] = scaler
    
    return scalers


def normalize_with_independent_scalers(
    df: pd.DataFrame,
    scalers: Dict[str, StandardScaler]
) -> pd.DataFrame:
    """
    Normalize dataframe using independent scalers per column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to normalize
    scalers : dict
        Dictionary of scalers (from create_independent_scalers)
    
    Returns
    -------
    pd.DataFrame
        Normalized dataframe with same structure as input
    
    Raises
    ------
    ValueError
        If df has columns not in scalers dict
    
    Examples
    --------
    >>> # Training
    >>> scalers_X = create_independent_scalers(X_train)
    >>> X_train_norm = normalize_with_independent_scalers(X_train, scalers_X)
    
    >>> # Test (using same scalers)
    >>> X_test_norm = normalize_with_independent_scalers(X_test, scalers_X)
    """
    df_norm = pd.DataFrame(index=df.index, columns=df.columns)
    
    for col in df.columns:
        if col not in scalers:
            raise ValueError(
                f"Column '{col}' not found in scalers. "
                f"Available columns: {list(scalers.keys())}"
            )
        
        df_norm[col] = scalers[col].transform(df[[col]])
    
    return df_norm


def denormalize_with_independent_scalers(
    df_norm: pd.DataFrame,
    scalers: Dict[str, StandardScaler]
) -> pd.DataFrame:
    """
    Denormalize dataframe using independent scalers.
    
    Inverse operation of normalize_with_independent_scalers.
    
    Parameters
    ----------
    df_norm : pd.DataFrame
        Normalized data
    scalers : dict
        Dictionary of scalers used for normalization
    
    Returns
    -------
    pd.DataFrame
        Denormalized dataframe in original scale
    
    Examples
    --------
    >>> # Predictions are in normalized space
    >>> y_pred_norm = model.predict(X_test_norm)
    
    >>> # Convert back to original scale
    >>> y_pred = denormalize_with_independent_scalers(y_pred_norm, scalers_y)
    """
    df_denorm = pd.DataFrame(index=df_norm.index, columns=df_norm.columns)
    
    for col in df_norm.columns:
        if col not in scalers:
            raise ValueError(f"Column '{col}' not found in scalers")
        
        df_denorm[col] = scalers[col].inverse_transform(df_norm[[col]])
    
    return df_denorm


def fit_and_transform_independent(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.DataFrame] = None,
    scaler_type: str = 'standard'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    Complete workflow: create scalers, normalize train, optionally normalize test.
    
    Convenience function that:
    1. Creates independent scalers from training data
    2. Normalizes training data
    3. Normalizes test data (if provided) using same scalers
    4. Returns normalized data and scalers
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.DataFrame
        Training targets
    X_test : pd.DataFrame, optional
        Test features
    y_test : pd.DataFrame, optional
        Test targets
    scaler_type : str, default='standard'
        Type of scaler ('standard', 'minmax', 'robust')
    
    Returns
    -------
    X_train_norm : pd.DataFrame
        Normalized training features
    y_train_norm : pd.DataFrame
        Normalized training targets
    scalers_X : dict
        Feature scalers
    scalers_y : dict
        Target scalers
    
    If X_test and y_test provided, also returns:
    X_test_norm : pd.DataFrame
    y_test_norm : pd.DataFrame
    
    Examples
    --------
    >>> # Training only
    >>> X_tr_n, y_tr_n, sc_X, sc_y = fit_and_transform_independent(
    ...     X_train, y_train
    ... )
    
    >>> # Training and test
    >>> results = fit_and_transform_independent(
    ...     X_train, y_train, X_test, y_test
    ... )
    >>> X_tr_n, y_tr_n, sc_X, sc_y, X_te_n, y_te_n = results
    """
    # Create scalers from training data
    scalers_X = create_independent_scalers(X_train, scaler_type=scaler_type)
    scalers_y = create_independent_scalers(y_train, scaler_type=scaler_type)
    
    # Normalize training data
    X_train_norm = normalize_with_independent_scalers(X_train, scalers_X)
    y_train_norm = normalize_with_independent_scalers(y_train, scalers_y)
    
    # Normalize test data if provided
    if X_test is not None and y_test is not None:
        X_test_norm = normalize_with_independent_scalers(X_test, scalers_X)
        y_test_norm = normalize_with_independent_scalers(y_test, scalers_y)
        
        return (
            X_train_norm, y_train_norm, scalers_X, scalers_y,
            X_test_norm, y_test_norm
        )
    else:
        return X_train_norm, y_train_norm, scalers_X, scalers_y


def save_scalers(
    scalers_X: Dict[str, StandardScaler],
    scalers_y: Dict[str, StandardScaler],
    output_dir: Union[str, Path],
    prefix: str = ''
) -> None:
    """
    Save scaler dictionaries to disk.
    
    Parameters
    ----------
    scalers_X : dict
        Feature scalers
    scalers_y : dict
        Target scalers
    output_dir : str or Path
        Directory to save scalers
    prefix : str, default=''
        Optional prefix for filenames
    
    Examples
    --------
    >>> save_scalers(scalers_X, scalers_y, 'models/scalers/')
    >>> # Creates scalers_X.pkl and scalers_y.pkl
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_path = output_dir / f'{prefix}scalers_X.pkl'
    y_path = output_dir / f'{prefix}scalers_y.pkl'
    
    joblib.dump(scalers_X, X_path)
    joblib.dump(scalers_y, y_path)
    
    print(f"✅ Scalers saved:")
    print(f"   X scalers ({len(scalers_X)} columns): {X_path}")
    print(f"   y scalers ({len(scalers_y)} columns): {y_path}")


def load_scalers(
    input_dir: Union[str, Path],
    prefix: str = ''
) -> Tuple[Dict, Dict]:
    """
    Load scaler dictionaries from disk.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing saved scalers
    prefix : str, default=''
        Optional prefix used when saving
    
    Returns
    -------
    scalers_X : dict
        Feature scalers
    scalers_y : dict
        Target scalers
    
    Examples
    --------
    >>> scalers_X, scalers_y = load_scalers('models/scalers/')
    >>> X_norm = normalize_with_independent_scalers(X, scalers_X)
    """
    input_dir = Path(input_dir)
    
    X_path = input_dir / f'{prefix}scalers_X.pkl'
    y_path = input_dir / f'{prefix}scalers_y.pkl'
    
    scalers_X = joblib.load(X_path)
    scalers_y = joblib.load(y_path)
    
    print(f"✅ Scalers loaded:")
    print(f"   X scalers: {len(scalers_X)} columns")
    print(f"   y scalers: {len(scalers_y)} columns")
    
    return scalers_X, scalers_y


def get_scaler_statistics(
    scalers: Dict[str, StandardScaler]
) -> pd.DataFrame:
    """
    Extract statistics from scalers for inspection.
    
    Parameters
    ----------
    scalers : dict
        Dictionary of fitted scalers
    
    Returns
    -------
    pd.DataFrame
        Dataframe with scaler parameters (mean, std, scale, etc.)
        depending on scaler type
    
    Examples
    --------
    >>> stats = get_scaler_statistics(scalers_X)
    >>> print(stats)
    >>> # Shows mean, std for each feature
    """
    stats_data = []
    
    for col_name, scaler in scalers.items():
        entry = {'column': col_name}
        
        # StandardScaler attributes
        if hasattr(scaler, 'mean_'):
            entry['mean'] = scaler.mean_[0]
            entry['std'] = scaler.scale_[0]
        
        # MinMaxScaler attributes
        if hasattr(scaler, 'min_'):
            entry['data_min'] = scaler.data_min_[0]
            entry['data_max'] = scaler.data_max_[0]
            entry['data_range'] = scaler.data_range_[0]
        
        # RobustScaler attributes
        if hasattr(scaler, 'center_'):
            entry['center'] = scaler.center_[0]
            entry['scale'] = scaler.scale_[0]
        
        stats_data.append(entry)
    
    return pd.DataFrame(stats_data)
