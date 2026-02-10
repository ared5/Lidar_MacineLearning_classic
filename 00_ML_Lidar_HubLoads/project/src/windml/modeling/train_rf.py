"""
Random Forest Training Module

Functions for training Random Forest models for multi-output regression.
Random Forest is an ensemble method that builds multiple decision trees
and averages their predictions.

Author: ML_Lidar_HubLoads Project
Date: 2026
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Optional, Tuple
import joblib
from pathlib import Path
import time


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: Optional[Dict] = None,
    verbose: bool = True
) -> RandomForestRegressor:
    """
    Train Random Forest model for multi-output regression.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.DataFrame
        Training targets
    params : dict, optional
        Random Forest parameters. If None, uses defaults.
        Common parameters:
        - n_estimators: Number of trees (default 100)
        - max_depth: Maximum tree depth (default None)
        - min_samples_split: Min samples to split node (default 2)
        - min_samples_leaf: Min samples in leaf (default 1)
        - max_features: Features per split (default 'sqrt')
        - oob_score: Whether to compute OOB score (default False)
        - n_jobs: Number of parallel jobs (default -1, all cores)
        - random_state: Random seed (default 42)
    verbose : bool, default=True
        Whether to print training progress
    
    Returns
    -------
    RandomForestRegressor
        Trained Random Forest model
    
    Examples
    --------
    >>> params = {
    ...     'n_estimators': 200,
    ...     'max_depth': 20,
    ...     'oob_score': True
    ... }
    >>> model = train_random_forest(X_train, y_train, params)
    """
    # Default parameters
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'oob_score': True,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 1 if verbose else 0
        }
    
    if verbose:
        print("="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        print(f"\n📊 Data:")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Targets: {y_train.shape[1]}")
        print(f"   Samples: {X_train.shape[0]:,}")
        
        print(f"\n⚙️  Parameters:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        print(f"\n⏳ Training...")
    
    start_time = time.time()
    
    # Create and train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\n✅ Training complete!")
        print(f"   Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        if params.get('oob_score', False):
            print(f"   OOB Score: {model.oob_score_:.4f}")
        
        print("="*70)
    
    return model


def evaluate_random_forest(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.DataFrame,
    dataset_name: str = 'Dataset'
) -> pd.DataFrame:
    """
    Evaluate Random Forest model and return metrics per target.
    
    Parameters
    ----------
    model : RandomForestRegressor
        Trained model
    X : pd.DataFrame
        Features for evaluation
    y : pd.DataFrame
        True targets
    dataset_name : str, default='Dataset'
        Name for display (e.g., 'Train', 'Test')
    
    Returns
    -------
    pd.DataFrame
        Metrics dataframe with columns: Target, RMSE, MAE, R2
    
    Examples
    --------
    >>> metrics_train = evaluate_random_forest(model, X_train, y_train, 'Train')
    >>> metrics_test = evaluate_random_forest(model, X_test, y_test, 'Test')
    >>> print(metrics_test)
    """
    # Predictions
    y_pred = model.predict(X)
    
    # Calculate metrics for each target
    metrics_data = []
    
    target_names = y.columns if isinstance(y, pd.DataFrame) else [f'Target_{i}' for i in range(y.shape[1])]
    
    for i, target_name in enumerate(target_names):
        if isinstance(y, pd.DataFrame):
            y_true = y.iloc[:, i]
            y_p = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        else:
            y_true = y[:, i]
            y_p = y_pred[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        mae = mean_absolute_error(y_true, y_p)
        r2 = r2_score(y_true, y_p)
        
        metrics_data.append({
            'Dataset': dataset_name,
            'Target': target_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and rank feature importances.
    
    Parameters
    ----------
    model : RandomForestRegressor
        Trained Random Forest model
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to return
    
    Returns
    -------
    pd.DataFrame
        Feature importance dataframe sorted by importance
    
    Examples
    --------
    >>> importance = get_feature_importance(model, X_train.columns, top_n=15)
    >>> print(importance)
    """
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df = importance_df.head(top_n).reset_index(drop=True)
    
    return importance_df


def save_random_forest(
    model: RandomForestRegressor,
    output_path: Union[str, Path],
    metadata: Optional[Dict] = None
) -> None:
    """
    Save trained Random Forest model to disk.
    
    Parameters
    ----------
    model : RandomForestRegressor
        Trained model to save
    output_path : str or Path
        Path for saving the model
    metadata : dict, optional
        Additional metadata to save alongside model
    
    Examples
    --------
    >>> save_random_forest(
    ...     model,
    ...     'models/rf_model.pkl',
    ...     metadata={'params': params, 'train_date': '2026-01-01'}
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Package model with metadata
    model_package = {
        'model': model,
        'metadata': metadata or {}
    }
    
    joblib.dump(model_package, output_path)
    print(f"✅ Random Forest model saved: {output_path}")


def load_random_forest(
    model_path: Union[str, Path]
) -> Tuple[RandomForestRegressor, Dict]:
    """
    Load trained Random Forest model from disk.
    
    Parameters
    ----------
    model_path : str or Path
        Path to saved model
    
    Returns
    -------
    model : RandomForestRegressor
        Loaded model
    metadata : dict
        Model metadata
    
    Examples
    --------
    >>> model, metadata = load_random_forest('models/rf_model.pkl')
    >>> print(f"Model trained on: {metadata.get('train_date')}")
    """
    model_package = joblib.load(model_path)
    
    if isinstance(model_package, dict):
        model = model_package['model']
        metadata = model_package.get('metadata', {})
    else:
        # Backward compatibility: model saved directly
        model = model_package
        metadata = {}
    
    print(f"✅ Random Forest model loaded: {model_path}")
    
    return model, metadata
