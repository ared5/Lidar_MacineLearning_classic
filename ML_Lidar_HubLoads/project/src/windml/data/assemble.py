"""
Data Assembly Module

Functions to combine multiple CSV files into a single complete dataset.
Handles memory-efficient loading, type optimization, and progress tracking.

Author: ML_Lidar_HubLoads Project
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import gc


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting data types.
    
    - float64 → float32 (saves 50% memory)
    - int64 → int32/int16 when possible
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    
    Returns
    -------
    pd.DataFrame
        Optimized dataframe
    
    Notes
    -----
    Typically reduces memory usage by 40-60% with minimal precision loss
    for machine learning applications.
    """
    df = df.copy()
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Convert float64 to float32
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        
        # Convert int64 to smaller int types
        elif col_type == 'int64':
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
    
    return df


def assemble_csvs(
    csv_folder: Union[str, Path],
    output_path: Union[str, Path],
    pattern: str = "*.csv",
    batch_size: int = 10,
    optimize_memory: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Assemble multiple CSV files into a single dataset.
    
    Loads CSVs in batches to manage memory efficiently, optionally optimizes
    data types, and saves the complete dataset.
    
    Parameters
    ----------
    csv_folder : str or Path
        Directory containing CSV files to combine
    output_path : str or Path
        Path for the output combined CSV file
    pattern : str, default="*.csv"
        Glob pattern to match CSV files (e.g., "*.csv", "DLC*.csv")
    batch_size : int, default=10
        Number of files to load at once before concatenating
    optimize_memory : bool, default=True
        Whether to optimize data types to reduce memory usage
    verbose : bool, default=True
        Whether to print progress information
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe containing all CSV data
    
    Examples
    --------
    >>> df_complete = assemble_csvs(
    ...     csv_folder='data/processed',
    ...     output_path='data/complete_dataset.csv',
    ...     batch_size=10
    ... )
    >>> print(f"Complete dataset: {df_complete.shape}")
    
    Notes
    -----
    For very large datasets (>1M rows), consider using chunked reading
    or Dask for out-of-core processing.
    """
    csv_folder = Path(csv_folder)
    output_path = Path(output_path)
    
    # Get list of CSV files
    csv_files = sorted(csv_folder.glob(pattern))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_folder} with pattern '{pattern}'")
    
    if verbose:
        print("="*70)
        print("ASSEMBLING CSV FILES")
        print("="*70)
        print(f"\nSource folder: {csv_folder}")
        print(f"Files found: {len(csv_files)}")
        print(f"Output: {output_path}")
        print(f"Batch size: {batch_size}")
        print("="*70)
    
    # Storage for batches
    all_batches = []
    loaded_count = 0
    failed_count = 0
    total_rows = 0
    memory_saved_total = 0
    
    # Process in batches
    num_batches = (len(csv_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(csv_files))
        batch_files = csv_files[start_idx:end_idx]
        
        if verbose:
            print(f"\nBatch {batch_idx + 1}/{num_batches}:")
        
        batch_dfs = []
        
        for csv_file in batch_files:
            try:
                # Load CSV
                df_temp = pd.read_csv(csv_file)
                rows = len(df_temp)
                
                # Optimize memory if requested
                if optimize_memory:
                    mem_before = df_temp.memory_usage(deep=True).sum() / 1024**2
                    df_temp = optimize_dataframe_dtypes(df_temp)
                    mem_after = df_temp.memory_usage(deep=True).sum() / 1024**2
                    mem_saved = mem_before - mem_after
                    memory_saved_total += mem_saved
                    
                    if verbose:
                        print(f"  ✓ {csv_file.name}: {rows:,} rows, "
                              f"{len(df_temp.columns)} cols, "
                              f"saved {mem_saved:.1f} MB")
                else:
                    if verbose:
                        print(f"  ✓ {csv_file.name}: {rows:,} rows, {len(df_temp.columns)} cols")
                
                batch_dfs.append(df_temp)
                loaded_count += 1
                total_rows += rows
                
            except Exception as e:
                if verbose:
                    print(f"  ✗ {csv_file.name}: ERROR - {str(e)}")
                failed_count += 1
        
        # Concatenate batch
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            all_batches.append(batch_df)
            
            # Free memory
            del batch_dfs
            gc.collect()
    
    # Concatenate all batches
    if verbose:
        print(f"\n{'='*70}")
        print("Combining all batches...")
    
    df_complete = pd.concat(all_batches, ignore_index=True)
    
    # Free memory
    del all_batches
    gc.collect()
    
    # Save to file
    if verbose:
        print(f"Saving to {output_path.name}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_complete.to_csv(output_path, index=False)
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("✅ ASSEMBLY COMPLETE")
        print("="*70)
        print(f"\nFiles loaded: {loaded_count}/{len(csv_files)}")
        if failed_count > 0:
            print(f"Files failed: {failed_count}")
        print(f"Total rows: {total_rows:,}")
        print(f"Total columns: {len(df_complete.columns)}")
        print(f"Final shape: {df_complete.shape}")
        
        if optimize_memory:
            mem_usage = df_complete.memory_usage(deep=True).sum() / 1024**2
            print(f"\nMemory usage: {mem_usage:.1f} MB")
            print(f"Memory saved: {memory_saved_total:.1f} MB")
        
        print(f"\nOutput saved to: {output_path}")
        print("="*70)
    
    return df_complete


def load_or_assemble_dataset(
    csv_folder: Union[str, Path],
    output_path: Union[str, Path],
    force_rebuild: bool = False,
    **assemble_kwargs
) -> pd.DataFrame:
    """
    Load existing assembled dataset or create it if it doesn't exist.
    
    Convenience function that checks if the complete dataset file exists.
    If it does, loads it. If not (or if force_rebuild=True), assembles
    from individual CSVs.
    
    Parameters
    ----------
    csv_folder : str or Path
        Directory containing individual CSV files
    output_path : str or Path
        Path to complete dataset CSV
    force_rebuild : bool, default=False
        If True, rebuilds dataset even if file exists
    **assemble_kwargs
        Additional arguments passed to assemble_csvs()
    
    Returns
    -------
    pd.DataFrame
        Complete dataset
    
    Examples
    --------
    >>> # First time: assembles from CSVs
    >>> df = load_or_assemble_dataset('data/processed', 'data/complete.csv')
    
    >>> # Subsequent times: loads from complete.csv (faster)
    >>> df = load_or_assemble_dataset('data/processed', 'data/complete.csv')
    
    >>> # Force rebuild
    >>> df = load_or_assemble_dataset(
    ...     'data/processed', 
    ...     'data/complete.csv',
    ...     force_rebuild=True
    ... )
    """
    output_path = Path(output_path)
    
    if output_path.exists() and not force_rebuild:
        print(f"✅ Loading existing dataset: {output_path.name}")
        df = pd.read_csv(output_path)
        print(f"   Shape: {df.shape}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return df
    else:
        if force_rebuild:
            print(f"🔄 Force rebuild requested. Assembling from CSVs...")
        else:
            print(f"📁 Dataset not found. Assembling from CSVs...")
        
        return assemble_csvs(
            csv_folder=csv_folder,
            output_path=output_path,
            **assemble_kwargs
        )


def verify_assembled_dataset(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    check_nans: bool = True,
    check_duplicates: bool = True
) -> dict:
    """
    Verify integrity of assembled dataset.
    
    Performs various checks to ensure the assembled dataset is valid:
    - Check for expected columns
    - Check for NaN values
    - Check for duplicate rows
    - Check data types
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to verify
    expected_columns : list of str, optional
        List of column names that must be present
    check_nans : bool, default=True
        Whether to check for NaN values
    check_duplicates : bool, default=True
        Whether to check for duplicate rows
    
    Returns
    -------
    dict
        Dictionary with verification results:
        - 'valid': bool, overall validation status
        - 'issues': list of str, any issues found
        - 'warnings': list of str, non-critical warnings
        - 'stats': dict with dataset statistics
    
    Examples
    --------
    >>> result = verify_assembled_dataset(df, expected_columns=['Time', 'My'])
    >>> if result['valid']:
    ...     print("Dataset OK!")
    ... else:
    ...     print(f"Issues: {result['issues']}")
    """
    issues = []
    warnings = []
    stats = {}
    
    # Check shape
    stats['n_rows'] = len(df)
    stats['n_cols'] = len(df.columns)
    
    # Check for expected columns
    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")
    
    # Check for NaN values
    if check_nans:
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            stats['nan_columns'] = len(nan_cols)
            stats['total_nans'] = nan_counts.sum()
            pct_nans = (nan_counts.sum() / df.size) * 100
            
            if pct_nans > 10:
                issues.append(f"High NaN percentage: {pct_nans:.2f}%")
            elif pct_nans > 1:
                warnings.append(f"Moderate NaN percentage: {pct_nans:.2f}%")
    
    # Check for duplicates
    if check_duplicates:
        n_duplicates = df.duplicated().sum()
        stats['n_duplicates'] = n_duplicates
        
        if n_duplicates > 0:
            pct_duplicates = (n_duplicates / len(df)) * 100
            warnings.append(
                f"Found {n_duplicates} duplicate rows ({pct_duplicates:.2f}%)"
            )
    
    # Check data types
    dtype_counts = df.dtypes.value_counts()
    stats['dtype_distribution'] = dtype_counts.to_dict()
    
    # Check memory usage
    stats['memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    # Overall validation
    valid = len(issues) == 0
    
    return {
        'valid': valid,
        'issues': issues,
        'warnings': warnings,
        'stats': stats
    }
