"""
Data Loader Module for Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
Provides functions to load, validate, inspect, and retrieve sample data.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

from config import (
    DATASET_PATH,
    TARGET_COLUMN,
    FEATURE_NAMES,
    LABEL_TO_NUMERIC,
    NUMERIC_TO_LABEL,
    ID_COLUMN,
)


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate the WDBC dataset against official specifications.
    
    Checks:
    - Exactly 569 rows.
    - Presence of 30 standard numerical features.
    - Presence of target column ('diagnosis').
    - Target contains valid classes (Benign / Malignant or 0 / 1 or B / M).
    """
    if len(df) != 569:
        return False, f"Expected 569 records, but found {len(df)}."

    if TARGET_COLUMN not in df.columns:
        return False, f"Missing required target column '{TARGET_COLUMN}'."

    missing_features = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_features:
        return False, f"Missing {len(missing_features)} required features: {missing_features[:3]}..."

    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    valid_target_sets = [
        {"B", "M"},
        {0, 1},
        {"Benign", "Malignant"},
    ]

    is_valid_target = any(unique_targets.issubset(s) or s.issubset(unique_targets) for s in valid_target_sets)
    if not is_valid_target:
        return False, f"Invalid target classes found: {unique_targets}"

    return True, "Dataset successfully validated (569 rows, 30 features, valid target)."


def load_dataset(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and clean the Wisconsin Diagnostic Breast Cancer dataset from disk.
    
    Args:
        dataset_path: Optional custom path. Defaults to config.DATASET_PATH.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with 30 features and target.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If validation fails.
    """
    path = dataset_path or DATASET_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please run 'python download_dataset.py' first."
        )

    df = pd.read_csv(path)

    # Drop ID column if present in model data
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Drop any 'Unnamed' columns created during CSV export
    unnamed_cols = [c for c in df.columns if "Unnamed" in c]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Validate structure
    is_valid, message = validate_dataset(df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {message}")

    return df


def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics and metadata for the dataset.
    """
    target_series = df[TARGET_COLUMN]

    # Universal robust matching across string/numeric dtypes
    benign_count = int(target_series.isin(["B", "Benign", 0, "0"]).sum())
    malignant_count = int(target_series.isin(["M", "Malignant", 1, "1"]).sum())

    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "total_records": len(df),
        "total_features": len([c for c in df.columns if c != TARGET_COLUMN]),
        "benign_cases": benign_count,
        "malignant_cases": malignant_count,
        "benign_percentage": round((benign_count / len(df)) * 100, 2),
        "malignant_percentage": round((malignant_count / len(df)) * 100, 2),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "features_list": [c for c in df.columns if c != TARGET_COLUMN],
    }


def get_sample_records(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Get representative real samples for Benign and Malignant tumors for UI presets.
    """
    target = df[TARGET_COLUMN]
    
    benign_mask = target.isin(["B", "Benign", 0, "0"])
    malignant_mask = target.isin(["M", "Malignant", 1, "1"])

    benign_idx = df[benign_mask].index[0]
    malignant_idx = df[malignant_mask].index[0]

    benign_sample = df.loc[benign_idx, FEATURE_NAMES].to_dict()
    malignant_sample = df.loc[malignant_idx, FEATURE_NAMES].to_dict()

    return {
        "Benign Sample (Patient #19)": {k: float(v) for k, v in benign_sample.items()},
        "Malignant Sample (Patient #0)": {k: float(v) for k, v in malignant_sample.items()},
    }
