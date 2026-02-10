"""
Time Series Split Module

Functions for splitting time series data into train/test sets while preserving
temporal structure. Implements series-aware splitting to avoid data leakage.

Key principle: Split by complete time series (simulations), not by random rows.

Author: ML_Lidar_HubLoads Project  
Date: 2026
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from pathlib import Path


def identify_time_series(
    df: pd.DataFrame,
    time_col: str = 'Time',
    reset_threshold: float = 0.0
) -> np.ndarray:
    """
    Identify individual time series in a concatenated dataset.
    
    Detects series boundaries by finding where the time column resets
    (decreases instead of increasing).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time column
    time_col : str, default='Time'
        Name of the time column
    reset_threshold : float, default=0.0
        Time decrease threshold for detecting series reset.
        If Time[i] < Time[i-1] - threshold, marks new series
    
    Returns
    -------
    np.ndarray
        Array of series IDs (integers) with same length as df.
        Each complete time series gets a unique ID (0, 1, 2, ...)
    
    Examples
    --------
    >>> series_id = identify_time_series(df, time_col='Time')
    >>> n_series = series_id.max() + 1
    >>> print(f"Found {n_series} time series")
    
    Notes
    -----
    This assumes time series are concatenated sequentially and time
    restarts at each new series. Common pattern when loading multiple
    simulation files into one dataset.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in dataframe")
    
    time_values = df[time_col].values
    series_id = np.zeros(len(df), dtype=np.int32)
    
    current_series = 0
    
    for i in range(1, len(time_values)):
        # Check if time decreased (series reset)
        if time_values[i] < (time_values[i-1] - reset_threshold):
            current_series += 1
        series_id[i] = current_series
    
    return series_id


def split_by_series(
    df: pd.DataFrame,
    test_series_ids: Union[List[int], np.ndarray],
    series_id_col: str = 'series_id'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe by series IDs.
    
    Assigns complete time series to either train or test set based on
    their series ID.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with series_id column
    test_series_ids : list or array
        List of series IDs to assign to test set
    series_id_col : str, default='series_id'
        Name of the column containing series IDs
    
    Returns
    -------
    df_train : pd.DataFrame
        Training set (all series NOT in test_series_ids)
    df_test : pd.DataFrame
        Test set (series in test_series_ids)
    
    Examples
    --------
    >>> # Put every 5th series in test
    >>> test_ids = np.arange(4, n_series, 5)  # 4, 9, 14, 19, ...
    >>> df_train, df_test = split_by_series(df, test_ids)
    
    >>> # Specific series to test
    >>> test_ids = [3, 7, 15, 22]
    >>> df_train, df_test = split_by_series(df, test_ids)
    """
    if series_id_col not in df.columns:
        raise ValueError(f"Series ID column '{series_id_col}' not found")
    
    # Create boolean masks
    test_mask = df[series_id_col].isin(test_series_ids)
    train_mask = ~test_mask
    
    # Split
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    # Reset indices
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    return df_train, df_test


def train_test_split_series(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_col: str = 'Time',
    test_series_ids: Optional[List[int]] = None,
    split_strategy: str = 'uniform',
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Split time series data into train and test sets.
    
    Implements series-aware splitting that preserves complete time series
    in either train or test (no series is split across both sets).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time series data
    test_size : float, default=0.2
        Fraction of series (not rows) to include in test set
    time_col : str, default='Time'
        Name of the time column for series identification
    test_series_ids : list of int, optional
        Explicit list of series IDs for test set.
        If provided, overrides test_size and split_strategy
    split_strategy : str, default='uniform'
        Strategy for selecting test series:
        - 'uniform': Every Nth series (e.g., every 5th for 20% test)
        - 'random': Random selection of series
        - 'end': Last N series
        - 'start': First N series
    random_state : int, optional
        Random seed for reproducible random splits
    
    Returns
    -------
    df_train : pd.DataFrame
        Training set
    df_test : pd.DataFrame
        Test set
    series_info : dict
        Dictionary with split information:
        - 'n_total_series': Total number of series
        - 'n_train_series': Number of series in train
        - 'n_test_series': Number of series in test
        - 'train_series_ids': Array of train series IDs
        - 'test_series_ids': Array of test series IDs
    
    Examples
    --------
    >>> # Uniform split: every 5th series to test (20%)
    >>> df_train, df_test, info = train_test_split_series(
    ...     df, test_size=0.2, split_strategy='uniform'
    ... )
    
    >>> # Random split with seed for reproducibility
    >>> df_train, df_test, info = train_test_split_series(
    ...     df, test_size=0.2, split_strategy='random', random_state=42
    ... )
    
    >>> # Explicit test series
    >>> df_train, df_test, info = train_test_split_series(
    ...     df, test_series_ids= [3, 7, 12, 18, 25]
    ... )
    """
    # Identify series
    series_id = identify_time_series(df, time_col=time_col)
    df['series_id'] = series_id
    
    n_series = series_id.max() + 1
    all_series_ids = np.arange(n_series)
    
    # Determine test series
    if test_series_ids is not None:
        # Use explicitamente provided test series
        test_ids = np.array(test_series_ids)
    else:
        # Select based on strategy and test_size
        n_test_series = max(1, int(n_series * test_size))
        
        if split_strategy == 'uniform':
            # Every Nth series
            step = max(1, int(1 / test_size))
            test_ids = all_series_ids[step-1::step][:n_test_series]
            
        elif split_strategy == 'random':
            # Random selection
            if random_state is not None:
                np.random.seed(random_state)
            test_ids = np.random.choice(
                all_series_ids, 
                size=n_test_series, 
                replace=False
            )
            test_ids = np.sort(test_ids)
            
        elif split_strategy == 'end':
            # Last N series
            test_ids = all_series_ids[-n_test_series:]
            
        elif split_strategy == 'start':
            # First N series
            test_ids = all_series_ids[:n_test_series]
            
        else:
            raise ValueError(
                f"Unknown split_strategy '{split_strategy}'. "
                f"Use 'uniform', 'random', 'end', or 'start'"
            )
    
    # Get train series IDs
    train_ids = np.array([sid for sid in all_series_ids if sid not in test_ids])
    
    # Split dataframe
    df_train, df_test = split_by_series(df, test_ids, series_id_col='series_id')
    
    # Create info dictionary
    series_info = {
        'n_total_series': n_series,
        'n_train_series': len(train_ids),
        'n_test_series': len(test_ids),
        'train_series_ids': train_ids,
        'test_series_ids': test_ids,
        'n_train_rows': len(df_train),
        'n_test_rows': len(df_test),
        'actual_test_fraction_series': len(test_ids) / n_series,
        'actual_test_fraction_rows': len(df_test) / len(df)
    }
    
    return df_train, df_test, series_info


def create_series_mapping(
    df: pd.DataFrame,
    time_col: str = 'Time',
    name_col: Optional[str] = 'Name_DLC',
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Create mapping between series IDs and simulation names.
    
    Useful for tracking which simulations ended up in train vs test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with time and optionally name columns
    time_col : str, default='Time'
        Time column for series identification
    name_col : str, optional
        Column containing simulation names (e.g., DLC names)
    output_path : str or Path, optional
        If provided, saves mapping to CSV file
    
    Returns
    -------
    pd.DataFrame
        Mapping dataframe with columns:
        - 'series_id': Series ID
        - 'name': Simulation name (if name_col provided)
        - 'n_rows': Number of rows in series
        - 'time_start': Start time
        - 'time_end': End time
        - 'duration': Duration in seconds
    
    Examples
    --------
    >>> mapping = create_series_mapping(
    ...     df, 
    ...     name_col='Name_DLC',
    ...     output_path='series_mapping.csv'
    ... )
    >>> print(mapping.head())
    """
    # Identify series
    series_id = identify_time_series(df, time_col=time_col)
    df['series_id'] = series_id
    
    n_series = series_id.max() + 1
    
    # Create mapping
    mapping_data = []
    
    for sid in range(n_series):
        mask = series_id == sid
        series_data = df[mask]
        
        mapping_entry = {
            'series_id': sid,
            'n_rows': len(series_data),
            'time_start': series_data[time_col].min(),
            'time_end': series_data[time_col].max(),
            'duration': series_data[time_col].max() - series_data[time_col].min()
        }
        
        # Add name if available
        if name_col and name_col in df.columns:
            names = series_data[name_col].unique()
            mapping_entry['name'] = names[0] if len(names) == 1 else str(names)
        
        mapping_data.append(mapping_entry)
    
    mapping_df = pd.DataFrame(mapping_data)
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(output_path, index=False)
        print(f"✅ Series mapping saved to: {output_path}")
    
    return mapping_df


def print_split_summary(series_info: dict) -> None:
    """
    Print formatted summary of train/test split.
    
    Parameters
    ----------
    series_info : dict
        Dictionary returned by train_test_split_series()
    
    Examples
    --------
    >>> df_train, df_test, info = train_test_split_series(df)
    >>> print_split_summary(info)
    """
    print("="*70)
    print("TRAIN/TEST SPLIT SUMMARY")
    print("="*70)
    
    print(f"\n📊 SERIES DISTRIBUTION:")
    print(f"   Total series:        {series_info['n_total_series']}")
    print(f"   Train series:        {series_info['n_train_series']} "
          f"({series_info['actual_test_fraction_series']*100:.1f}%)")
    print(f"   Test series:         {series_info['n_test_series']} "
          f"({(1-series_info['actual_test_fraction_series'])*100:.1f}%)")
    
    print(f"\n📊 ROW DISTRIBUTION:")
    print(f"   Total rows:          {series_info['n_train_rows'] + series_info['n_test_rows']:,}")
    print(f"   Train rows:          {series_info['n_train_rows']:,} "
          f"({(1-series_info['actual_test_fraction_rows'])*100:.1f}%)")
    print(f"   Test rows:           {series_info['n_test_rows']:,} "
          f"({series_info['actual_test_fraction_rows']*100:.1f}%)")
    
    print(f"\n📝 SERIES IDS:")
    print(f"   Train series IDs:    {list(series_info['train_series_ids'][:10])}...")
    print(f"   Test series IDs:     {list(series_info['test_series_ids'])}")
    
    print("\n" + "="*70)
