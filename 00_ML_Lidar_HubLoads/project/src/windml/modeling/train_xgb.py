"""
XGBoost Training Module - Train XGBoost models for wind turbine load prediction.

This module provides functions to train XGBoost models using either MultiOutputRegressor
or individual models per target with early stopping.

Author: Wind ML Team
Date: February 2026
"""

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not available. Install with: pip install xgboost")


def train_xgboost_multioutput(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: Dict[str, Any],
    model_name: str = "xgboost",
    save_path: Optional[Path] = None
) -> MultiOutputRegressor:
    """
    Train an XGBoost model using MultiOutputRegressor.
    
    Args:
        X_train: Training features
        y_train: Training targets (multiple outputs)
        params: XGBoost parameters dictionary
        model_name: Name for saving the model
        save_path: Directory to save the trained model
        
    Returns:
        Trained MultiOutputRegressor model
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is required for this function")
    
    print("=" * 70)
    print("TRAINING XGBOOST MULTIOUTPUT MODEL")
    print("=" * 70)
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  Targets: {list(y_train.columns)}")
    
    print(f"\nModel parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create base XGBoost model
    xgb_model = xgb.XGBRegressor(**params)
    
    # Wrap with MultiOutputRegressor
    model = MultiOutputRegressor(xgb_model, n_jobs=-1)
    
    # Train
    print(f"\nTraining...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Training completed in {training_time:.2f}s")
    
    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / f"{model_name}.pkl"
        joblib.dump(model, model_file)
        print(f"✓ Model saved: {model_file}")
    
    return model


def train_xgboost_individual(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    params: Dict[str, Any],
    save_path: Optional[Path] = None
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Train individual XGBoost models for each target with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training targets (multiple outputs)
        X_val: Validation features
        y_val: Validation targets
        params: XGBoost parameters (must include early_stopping_rounds)
        save_path: Directory to save the trained models
        
    Returns:
        Tuple of (models_dict, metrics_dict)
        - models_dict: Dictionary {target_name: trained_model}
        - metrics_dict: Dictionary {target_name: {metric: value}}
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is required for this function")
    
    print("=" * 70)
    print("TRAINING INDIVIDUAL XGBOOST MODELS WITH EARLY STOPPING")
    print("=" * 70)
    
    targets = y_train.columns.tolist()
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"  y_train: {y_train.shape}, y_val: {y_val.shape}")
    print(f"  Targets: {targets}")
    
    print(f "\nModel parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    models = {}
    best_iters = {}
    metrics_val = {}
    training_times = {}
    
    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] Training model for: {target}")
        print(f"  {'─'*60}")
        
        start_time = time.time()
        
        # Create individual model
        model = xgb.XGBRegressor(**params)
        
        # Train with early stopping
        model.fit(
            X_train, y_train[target],
            eval_set=[(X_val, y_val[target])],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Save model and metadata
        models[target] = model
        best_iters[target] = getattr(model, 'best_iteration', None)
        training_times[target] = training_time
        
        # Evaluate on validation
        pred_val = model.predict(X_val)
        
        metrics_val[target] = {
            'R2': r2_score(y_val[target], pred_val),
            'RMSE': np.sqrt(mean_squared_error(y_val[target], pred_val)),
            'MAE': mean_absolute_error(y_val[target], pred_val),
            'best_iter': best_iters[target]
        }
        
        print(f"  • Time: {training_time:.2f}s")
        print(f"  • Best iteration: {metrics_val[target]['best_iter']}")
        print(f"  • R² (val): {metrics_val[target]['R2']:.6f}")
        print(f"  • RMSE (val): {metrics_val[target]['RMSE']:.4f} kNm")
        print(f"  • MAE (val): {metrics_val[target]['MAE']:.4f} kNm")
    
    total_training_time = sum(training_times.values())
    print(f"\n✓ All models trained in {total_training_time:.2f}s")
    
    # Save models if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for target in targets:
            model_file = save_path / f"xgboost_individual_{target}.pkl"
            joblib.dump(models[target], model_file)
        
        # Save complete dict
        models_dict_file = save_path / "xgboost_individual_models.pkl"
        joblib.dump(models, models_dict_file)
        
        # Save metadata
        metadata = {
            'targets': targets,
            'best_iters': best_iters,
            'metrics_val': metrics_val,
            'training_times': training_times,
            'params': params
        }
        metadata_file = save_path / "training_metadata.pkl"
        joblib.dump(metadata, metadata_file)
        
        print(f"\n✓ Models saved to: {save_path}")
    
    return models, metrics_val


def predict_xgboost(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "multioutput"
) -> pd.DataFrame:
    """
    Make predictions with an XGBoost model.
    
    Args:
        model: Trained model (MultiOutputRegressor or dict of individual models)
        X: Features for prediction
        model_type: Type of model ("multioutput" or "individual")
        
    Returns:
        DataFrame with predictions
    """
    if model_type == "multioutput":
        predictions = model.predict(X)
        # Get target names from estimators if available
        if hasattr(model, 'estimators_'):
            # Try to infer target names
            target_names = [f"target_{i}" for i in range(predictions.shape[1])]
        else:
            target_names = [f"target_{i}" for i in range(predictions.shape[1])]
        
        pred_df = pd.DataFrame(predictions, index=X.index, columns=target_names)
        
    elif model_type == "individual":
        # Assume model is a dict {target: model}
        pred_df = pd.DataFrame(index=X.index)
        for target, target_model in model.items():
            pred_df[target] = target_model.predict(X)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return pred_df
