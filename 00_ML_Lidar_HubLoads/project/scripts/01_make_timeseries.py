"""
Script 01: Generate time series CSV files from Bladed simulations.

This script reads Bladed simulation files (.$ files) and creates CSV files
with time series data for selected variables.

Usage:
    python 01_make_timeseries.py
    
Configuration:
    Edit the variables below to customize:
    - LOADPATH: Directory with Bladed simulations
    - FILE_NAMES: List of file patterns to process
    - VAR_DICTS: Variables to extract
    - RESULTSPATH: Output directory for CSVs

Author: Wind ML Team
Date: February 2026
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from windml.config.settings import get_config
from windml.data.bladed_io import load_bladed_data

# ============================================================================
# CONFIGURATION
# ============================================================================

# Bladed simulations directory
LOADPATH = r"U:\Studies\437_lidar_VLOS_IPC\Outputs\OPT3_PassiveYaw\DLC13"

# File patterns to process
FILE_NAMES = [
    "0020_DLC13_090_000",
    "0021_DLC13_090_000",
    "0022_DLC13_090_000",
    "0023_DLC13_090_000",
    "0024_DLC13_090_000",
    # Add more files as needed
]

# Variables to extract (organized by signal type)
VAR_DICTS = {
    "Hub_rotating": {
        "Hub_rotating": [
            "Blade root 1 My",
            "Blade root 2 My"
        ]
    },
    "Pitch_actuator": {
        "Pitch_actuator": [
            "Blade 1 pitch angle",
            "Blade 2 pitch angle",
        ]
    },
    "Drive_train": {
        "Drive_train": ["Rotor azimuth angle"]
    },
    "Summary": {
        "Summary": ["Rotor speed"]
    },
    "External_controller": {
        "External_controller": [
            "LAC_VLOS_BEAM0_RANGE5",
            "LAC_VLOS_BEAM1_RANGE5",
            "LAC_VLOS_BEAM2_RANGE5",
            "LAC_VLOS_BEAM3_RANGE5",
            "LAC_VLOS_BEAM4_RANGE5",
            "LAC_VLOS_BEAM5_RANGE5",
            "LAC_VLOS_BEAM6_RANGE5",
            "LAC_VLOS_BEAM7_RANGE5",
            "OPER_MEAS_YAWERROR",
        ]
    }
}

# Radial positions for Aero signals (if needed)
AERO_POSITIONS = [0.0, 6.0, 18.0, 30.0, 46.0, 59.0, 68.25]

# Add units row to CSV
ADD_UNITS = False


def main():
    """Main execution function."""
    print("=" * 80)
    print("SCRIPT 01: GENERATE TIME SERIES FROM BLADED SIMULATIONS")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    
    # Get output path from config
    resultspath = str(config.get_path('data', 'raw'))
    
    print(f"\nConfiguration:")
    print(f"  Input directory:  {LOADPATH}")
    print(f"  Output directory: {resultspath}")
    print(f"  Files to process: {len(FILE_NAMES)}")
    print(f"  Variable groups:  {len(VAR_DICTS)}")
    
    # Load Bladed data and create CSVs
    dataframes = load_bladed_data(
        loadpath=LOADPATH,
        file_names=FILE_NAMES,
        var_dicts=VAR_DICTS,
        resultspath=resultspath,
        add_units=ADD_UNITS,
        aero_positions=AERO_POSITIONS if any('Aero' in k for k in VAR_DICTS.keys()) else None
    )
    
    print(f"\n{'='*80}")
    print("SCRIPT 01 COMPLETED")
    print("="*80)
    print(f"\n✓ Created {len(dataframes)} CSV files in: {resultspath}")
    print(f"✓ Next step: Run 02_make_features.py to create engineered features")


if __name__ == "__main__":
    main()
