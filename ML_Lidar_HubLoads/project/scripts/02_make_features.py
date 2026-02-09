"""
Script 02: Create engineered features from raw time series data.

This script applies feature engineering transformations to raw CSV files:
- VLOS lags (time-shifted wind measurements)
- Trigonometric components (sin/cos of angles)
- Coleman transform (0P, 1P, 2P frequency components)
- Additional statistical features

Usage:
    python 02_make_features.py
    
Configuration:
    Customize feature creation in configs/features.yaml

Author: Wind ML Team
Date: February 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from windml import (
    get_config,
    get_feature_config,
    create_frequency_components_1P_2P,
    create_vlos_lags,
    create_azimuth_components,
    create_yawerror_components
)


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to a DataFrame.
    
    Args:
        df: DataFrame with raw time series data
        
    Returns:
        DataFrame with engineered features added
    """
    print("\n" + "=" * 70)
    print("APPLYING FEATURE ENGINEERING")
    print("=" * 70)
    
    # Get feature configuration
    config = get_config()
    
    # 1. Trigonometric components
    if get_feature_config('engineered_features', 'trigonometric', 'enabled'):
        print("\n[1/4] Creating trigonometric components...")
        
        # Azimuth components
        if 'Rotor azimuth angle' in df.columns:
            df = create_azimuth_components(df)
        
        # Yaw error components
        if 'OPER_MEAS_YAWERROR' in df.columns:
            df = create_yawerror_components(df)
    
    # 2. VLOS lags
    if get_feature_config('engineered_features', 'vlos_lags', 'enabled'):
        print("\n[2/4] Creating VLOS lags...")
        
        lag_config = get_feature_config('engineered_features', 'vlos_lags')
        lag_seconds = lag_config.get('lag_seconds', [2, 5, 8, 11, 14, 17, 20, 23, 26])
        
        range_filter = lag_config.get('range_filter', {})
        if range_filter.get('enabled', False):
            range_values = range_filter.get('range_values', [5])
            df = create_vlos_lags(df, lag_seconds_list=lag_seconds, range_values=range_values)
        else:
            df = create_vlos_lags(df, lag_seconds_list=lag_seconds, include_all_ranges=True)
    
    # 3. Coleman transform (0P, 1P, 2P components)
    if get_feature_config('engineered_features', 'coleman_transform', 'enabled'):
        print("\n[3/4] Creating Coleman frequency components...")
        
        apply_filtering = get_feature_config('engineered_features', 'coleman_transform', 'apply_filtering')
        df = create_frequency_components_1P_2P(df, apply_filtering=apply_filtering)
    
    # 4. Additional features (placeholder for future extensions)
    print("\n[4/4] Additional features (if configured)...")
    print("   • No additional features configured")
    
    print("\n" + "=" * 70)
    print(f"FEATURE ENGINEERING COMPLETED")
    print(f"Final shape: {df.shape}")
    print("=" * 70)
    
    return df


def main():
    """Main execution function."""
    print("=" * 80)
    print("SCRIPT 02: CREATE ENGINEERED FEATURES")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    
    # Get paths
    raw_data_path = config.get_path('data', 'raw')
    processed_data_path = config.get_path('data', 'processed')
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nPaths:")
    print(f"  Input:  {raw_data_path}")
    print(f"  Output: {processed_data_path}")
    
    # Find all CSV files in raw data directory
    csv_files = list(raw_data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"\n✗ No CSV files found in {raw_data_path}")
        print(f"  Run 01_make_timeseries.py first to generate raw data")
        return
    
    print(f"\nFound {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(csv_files)}] Processing: {csv_file.name}")
        print("="*80)
        
        try:
            # Load CSV
            print(f"Loading data...")
            df = pd.read_csv(csv_file)
            print(f"  ✓ Loaded: {df.shape}")
            
            # Apply feature engineering
            df_processed = apply_feature_engineering(df.copy())
            
            # Save processed data
            output_file = processed_data_path / csv_file.name
            df_processed.to_csv(output_file, index=False)
            print(f"\n  ✓ Saved: {output_file}")
            print(f"    Shape: {df_processed.shape}")
            
        except Exception as e:
            print(f"\n  ✗ Error processing {csv_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("SCRIPT 02 COMPLETED")
    print("="*80)
    print(f"\n✓ Processed {len(csv_files)} files")
    print(f"✓ Output directory: {processed_data_path}")
    print(f"✓ Next step: Run 03_build_dataset.py to combine files into ML dataset")


if __name__ == "__main__":
    main()
