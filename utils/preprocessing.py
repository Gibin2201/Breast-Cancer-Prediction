"""
Preprocessing Module for Breast Cancer Prediction.
Handles target encoding, stratified splitting, and feature scaling pipelines.
Strictly ensures no data leakage by utilizing Scikit-learn Pipelines.
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from config import (
    TARGET_COLUMN,
    FEATURE_NAMES,
    LABEL_TO_NUMERIC,
    RANDOM_SEED,
    TEST_SIZE,
)


def encode_target(df: pd.DataFrame) -> pd.Series:
    """
    Encode diagnosis column into standard binary numeric format:
    0 = Benign
    1 = Malignant
    """
    target = df[TARGET_COLUMN].copy()

    mapping = {
        "B": 0,
        "Benign": 0,
        "benign": 0,
        "0": 0,
        0: 0,
        "M": 1,
        "Malignant": 1,
        "malignant": 1,
        "1": 1,
        1: 1,
    }

    # Map directly or strip whitespace
    mapped = target.map(mapping)
    if mapped.isnull().any():
        mapped = target.astype(str).str.strip().map(mapping)

    if mapped.isnull().any():
        unmapped = target[mapped.isnull()].unique()
        raise ValueError(f"Unrecognized target values found in diagnosis: {unmapped}")

    return mapped.astype(int)


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y with guaranteed feature ordering.
    """
    # Verify all 30 features exist
    for feature in FEATURE_NAMES:
        if feature not in df.columns:
            raise KeyError(f"Required feature '{feature}' is missing from DataFrame.")

    X = df[FEATURE_NAMES].copy()
    y = encode_target(df)

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split to preserve benign/malignant proportions.
    
    Args:
        X: Feature DataFrame
        y: Binary target Series
        test_size: Fraction of dataset for testing (default: 0.20)
        random_state: Reproducibility seed (default: 42)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def create_pipeline(
    classifier: BaseEstimator,
    use_scaler: bool = True,
) -> Pipeline:
    """
    Create a Scikit-learn Pipeline with optional StandardScaler to avoid data leakage.
    
    Args:
        classifier: Scikit-learn estimator instance
        use_scaler: Whether to precede classifier with StandardScaler (default True)
        
    Returns:
        Pipeline: Sklearn Pipeline object
    """
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))

    return Pipeline(steps=steps)


def validate_input_dict(input_data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
    """
    Validate user-submitted input dictionary containing all 30 tumor features.
    
    Returns:
        (is_valid, error_message, feature_array)
    """
    if len(input_data) != 30:
        return False, f"Expected exactly 30 features, but received {len(input_data)}.", None

    ordered_values = []
    for feature in FEATURE_NAMES:
        if feature not in input_data:
            return False, f"Missing required feature: '{feature}'", None
        
        val = input_data[feature]
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            return False, f"Feature '{feature}' has non-numeric value: {val}", None

        if val_float < 0:
            return False, f"Feature '{feature}' cannot be negative (got {val_float}).", None

        ordered_values.append(val_float)

    # Return as 2D array ready for pipeline input
    return True, None, np.array(ordered_values).reshape(1, -1)
