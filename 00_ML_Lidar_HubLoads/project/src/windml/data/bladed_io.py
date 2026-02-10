"""
Bladed I/O Module - Functions to read Bladed simulation files and export to CSV.

This module handles reading Bladed binary files (.$ files) and creating
time series CSV files with selected signals.

Author: Wind ML Team
Date: February 2026
"""

import os
import csv
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import postprocessbladed as pp


def create_timeseries_csv(
    bin_series: Dict,
    header_series: Dict,
    filenames: List[str],
    var_dict: Dict[str, List[str]],
    output_path: str,
    output_filename: str,
    add_units: bool = False,
    aero_positions: Optional[List[float]] = None
) -> str:
    """
    Create a CSV file with time series from Bladed binary files.
    
    This function replicates the behavior of timeseries_script.py from the notebook.
    
    Args:
        bin_series: Dictionary with binary data {file: {signal: data}}
        header_series: Headers read with pp.read_hdr_files()
        filenames: List of processed files
        var_dict: Dictionary with variables to extract {signal: [variables]}
        output_path: Path where to save the CSV
        output_filename: Filename (without extension)
        add_units: If True, add units row
        aero_positions: Radial positions for Aero signals (meters)
        
    Returns:
        Path to the created CSV file
    """
    print(f'Creating CSV for variables: {list(var_dict.keys())}')
    
    # Units dictionary (SI units)
    variable_units = {
        "Time": "s",
        "Rotor average longitudinal wind speed": "m/s",
        "Blade 1 Incident axial wind speed": "m/s",
        "Blade 2 Incident axial wind speed": "m/s",
        "Blade root 1 My": "kNm",
        "Blade root 2 My": "kNm",
        "Blade 1 pitch angle": "deg",
        "Blade 2 pitch angle": "deg",
        "Blade 1 pitch rate": "deg/s",
        "Blade 2 pitch rate": "deg/s",
        "Rotor azimuth angle": "deg",
        "Rotor speed": "rpm",
        "LAC_VLOS_BEAM0_RANGE5": "m/s",
        "LAC_VLOS_BEAM1_RANGE5": "m/s",
        "LAC_VLOS_BEAM2_RANGE5": "m/s",
        "LAC_VLOS_BEAM3_RANGE5": "m/s",
        "LAC_VLOS_BEAM4_RANGE5": "m/s",
        "LAC_VLOS_BEAM5_RANGE5": "m/s",
        "LAC_VLOS_BEAM6_RANGE5": "m/s",
        "LAC_VLOS_BEAM7_RANGE5": "m/s",
        "OPER_MEAS_YAWERROR": "deg"
    }
    
    # Get all variable names for header
    all_variables = []
    for signal in var_dict.keys():
        all_variables.extend(var_dict[signal])
    
    # Create header row
    header = ['Time'] + all_variables
    
    # Initialize data structure
    csv_data = [header]
    
    # Add units row if requested
    if add_units:
        units_row = []
        for var in header:
            if var in variable_units:
                units_row.append(variable_units[var])
            else:
                units_row.append("")
        csv_data.append(units_row)
    
    # Process each file
    for file in filenames:
        print(f'Processing file: {os.path.basename(file)}')
        
        # Get time vector
        time_data = None
        first_signal = list(var_dict.keys())[0]
        first_variable = var_dict[first_signal][0]
        
        try:
            if first_signal in bin_series[file]:
                if isinstance(bin_series[file][first_signal], np.ndarray):
                    signal_data = bin_series[file][first_signal]
                    
                    # Special handling for Aero signals
                    if first_signal in ("Aero_B1", "Aero_B2") and \
                       hasattr(signal_data.dtype, 'names') and signal_data.dtype.names:
                        base_name = signal_data.dtype.names[0]
                        arr = signal_data[base_name]
                        if arr.ndim == 2:
                            time_length = arr.shape[-1] if arr.shape[-1] >= arr.shape[0] else arr.shape[0]
                        else:
                            time_length = len(arr)
                        try:
                            dt = header_series[file][first_signal]['dtime']
                        except KeyError:
                            dt = 0.02
                        time_data = [i * dt for i in range(time_length)]
                    else:
                        # Generic handling
                        if hasattr(signal_data.dtype, 'names') and signal_data.dtype.names:
                            if first_variable in signal_data.dtype.names:
                                time_length = len(signal_data[first_variable])
                                try:
                                    dt = header_series[file][first_signal]['dtime']
                                except KeyError:
                                    dt = 0.02
                                time_data = [i * dt for i in range(time_length)]
                        else:
                            time_length = len(signal_data)
                            try:
                                dt = header_series[file][first_signal]['dtime']
                            except KeyError:
                                dt = 0.02
                            time_data = [i * dt for i in range(time_length)]
        except Exception as e:
            print(f'Error getting time data: {e}')
            continue
        
        if time_data is None:
            print(f"Warning: Could not determine time data for {file}")
            continue
        
        print(f'Processing {len(time_data)} time steps')
        
        # Process each time step
        for i in range(len(time_data)):
            row = [time_data[i]]
            
            # Add data for each variable
            for signal in var_dict.keys():
                for variable in var_dict[signal]:
                    try:
                        if signal in bin_series[file]:
                            signal_data = bin_series[file][signal]
                            
                            if isinstance(signal_data, np.ndarray):
                                # Special handling for Aero signals
                                if signal in ("Aero_B1", "Aero_B2") and aero_positions:
                                    m = re.search(r"at\s+([0-9]+(?:\.[0-9]+)?)m", variable)
                                    if m:
                                        try:
                                            pos = float(m.group(1))
                                            idx = None
                                            for j, p in enumerate(aero_positions):
                                                if abs(p - pos) < 1e-6:
                                                    idx = j
                                                    break
                                            if idx is None:
                                                idx = int(np.argmin([abs(p - pos) for p in aero_positions]))
                                            
                                            if hasattr(signal_data.dtype, 'names') and signal_data.dtype.names:
                                                base_name = signal_data.dtype.names[0]
                                                arr = signal_data[base_name]
                                            else:
                                                arr = signal_data
                                            
                                            if arr.ndim == 2:
                                                val = arr[idx, i]
                                            elif arr.ndim == 1:
                                                val = arr[i] if i < arr.shape[0] else np.nan
                                            else:
                                                val = np.nan
                                            row.append(float(val) if np.isfinite(val) else 0.0)
                                        except Exception as ex:
                                            row.append(0.0)
                                    else:
                                        row.append(0.0)
                                else:
                                    # Non-Aero arrays
                                    if hasattr(signal_data.dtype, 'names') and signal_data.dtype.names:
                                        if variable in signal_data.dtype.names and i < len(signal_data[variable]):
                                            row.append(float(signal_data[variable][i]))
                                        else:
                                            row.append(0.0)
                                    else:
                                        if signal_data.ndim == 1 and i < signal_data.shape[0]:
                                            row.append(float(signal_data[i]))
                                        else:
                                            row.append(0.0)
                            else:
                                row.append(0.0)
                        else:
                            row.append(0.0)
                    except Exception as e:
                        row.append(0.0)
            
            csv_data.append(row)
    
    # Write to CSV
    output_file = os.path.join(output_path, f"{output_filename}.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f'CSV file created: {output_file}')
    return output_file


def load_bladed_data(
    loadpath: str,
    file_names: List[str],
    var_dicts: Dict[str, Dict[str, List[str]]],
    resultspath: str,
    add_units: bool = False,
    aero_positions: Optional[List[float]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load Bladed simulation files and create CSV files.
    
    Args:
        loadpath: Directory with Bladed simulation files (.$ files)
        file_names: List of file name patterns to process
        var_dicts: Dictionary of variable dictionaries {dict_name: {signal: [vars]}}
        resultspath: Output directory for CSV files
        add_units: Add units row to CSV
        aero_positions: Radial positions for Aero signals
        
    Returns:
        Dictionary of DataFrames {filename: dataframe}
    """
    print("="*70)
    print("LOADING BLADED DATA")
    print("="*70)
    
    # Combine all variable dictionaries
    combined_var_dict = {}
    for dict_name, var_dict in var_dicts.items():
        combined_var_dict.update(var_dict)
    
    signal_list = list(combined_var_dict.keys())
    
    print(f"\nSearching for files in: {loadpath}")
    print(f"Patterns to match: {len(file_names)}")
    
    # Find all simulation files
    all_dlcnames = pp.read_dlcnames(loadpath, file_filter={"": 1})
    print(f"Total .$ files found: {len(all_dlcnames)}")
    
    # Filter files for each pattern
    all_matching_files = {}
    for file_name in file_names:
        matching = [f for f in all_dlcnames if file_name in os.path.basename(f)]
        all_matching_files[file_name] = matching
        if matching:
            print(f"  ✓ {file_name}: {len(matching)} file(s)")
        else:
            print(f"  ✗ {file_name}: No files found")
    
    # Process each file pattern
    all_dataframes = {}
    
    for file_name_pattern in file_names:
        current_files = all_matching_files.get(file_name_pattern, [])
        
        if not current_files:
            continue
        
        print(f"\nProcessing: {file_name_pattern}")
        
        try:
            # Read headers
            header_series = pp.read_hdr_files(current_files, signal_list)
            
            # Read binary data
            non_aero_signals = [s for s in signal_list if "Aero" not in s]
            aero_signals = [s for s in signal_list if "Aero" in s]
            
            bin_series = {f: {} for f in current_files}
            
            # Read non-Aero signals
            if non_aero_signals:
                non_aero_bin = pp.read_bin_files(current_files, header_series, non_aero_signals)
                for f in current_files:
                    if f in non_aero_bin:
                        bin_series[f].update(non_aero_bin[f])
            
            # Read Aero signals with specific positions
            if aero_signals and aero_positions:
                for aero_sig in aero_signals:
                    first_file = current_files[0]
                    hdr_vars = header_series[first_file][aero_sig].get('variab', [])
                    
                    base_var = None
                    for v in hdr_vars:
                        if 'incident' in v.lower() and 'axial' in v.lower() and 'wind speed' in v.lower():
                            base_var = v
                            break
                    
                    if base_var is None and hdr_vars:
                        base_var = hdr_vars[0]
                    
                    aero_bin = pp.read_bin_files(
                        current_files,
                        header_series,
                        [aero_sig],
                        var_list=[base_var],
                        reduced_mbr_list=aero_positions
                    )
                    
                    for f in current_files:
                        if f in aero_bin:
                            bin_series[f].update(aero_bin[f])
            
            # Create CSV
            output_filename = file_name_pattern
            csv_file = create_timeseries_csv(
                bin_series,
                header_series,
                current_files,
                combined_var_dict,
                resultspath,
                output_filename,
                add_units,
                aero_positions
            )
            
            # Load CSV as DataFrame
            df = pd.read_csv(csv_file)
            all_dataframes[file_name_pattern] = df
            print(f"✓ Loaded: {df.shape}")
            
        except Exception as e:
            print(f"✗ Error processing {file_name_pattern}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"COMPLETED: {len(all_dataframes)} datasets created")
    print("="*70)
    
    return all_dataframes
