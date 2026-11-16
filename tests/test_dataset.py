"""
Unit Tests for Dataset Loading, Integrity, and Preprocessing.
Verifies the WDBC dataset specifications (569 rows, 30 features, target classes).
"""

import pytest
import pandas as pd
from pathlib import Path

from config import DATASET_PATH, TARGET_COLUMN, FEATURE_NAMES
from utils.data_loader import (
    load_dataset,
    validate_dataset,
    get_dataset_metadata,
    get_sample_records,
)
from download_dataset import download_dataset


@pytest.fixture(scope="module")
def dataset():
    """Ensure dataset is present and load DataFrame."""
    if not DATASET_PATH.exists():
        download_dataset()
    return load_dataset()


def test_dataset_exists():
    """Verify that dataset file exists on disk."""
    if not DATASET_PATH.exists():
        download_dataset()
    assert DATASET_PATH.exists(), f"Dataset file missing at {DATASET_PATH}"


def test_dataset_record_count(dataset):
    """Verify that the dataset contains exactly 569 records."""
    assert len(dataset) == 569, f"Expected 569 records, found {len(dataset)}"


def test_dataset_feature_count(dataset):
    """Verify that all 30 standard numerical features are present."""
    for feat in FEATURE_NAMES:
        assert feat in dataset.columns, f"Required feature '{feat}' is missing"
    assert len(FEATURE_NAMES) == 30, "Feature list does not have exactly 30 features"


def test_dataset_target_column(dataset):
    """Verify target column exists and contains Benign and Malignant classes."""
    assert TARGET_COLUMN in dataset.columns, f"Target column '{TARGET_COLUMN}' missing"
    unique_vals = dataset[TARGET_COLUMN].dropna().unique()
    # Check if target values represent binary diagnosis
    has_b_m = any(val in ["B", "Benign", 0] for val in unique_vals) and any(
        val in ["M", "Malignant", 1] for val in unique_vals
    )
    assert has_b_m, f"Target column values unexpected: {unique_vals}"


def test_dataset_missing_values(dataset):
    """Verify that there are no missing values in the 30 numerical features."""
    null_counts = dataset[FEATURE_NAMES].isnull().sum().sum()
    assert null_counts == 0, f"Found {null_counts} null values in numerical features"


def test_dataset_metadata_generation(dataset):
    """Verify dataset metadata summary dictionary structure."""
    meta = get_dataset_metadata(dataset)
    assert meta["total_records"] == 569
    assert meta["total_features"] == 30
    assert meta["benign_cases"] + meta["malignant_cases"] == 569
    assert meta["missing_values"] == 0


def test_sample_records(dataset):
    """Verify that sample tumor records contain all 30 numerical features."""
    samples = get_sample_records(dataset)
    assert len(samples) >= 2, "Expected benign and malignant sample records"
    for name, sample_dict in samples.items():
        assert len(sample_dict) == 30
        for f in FEATURE_NAMES:
            assert f in sample_dict
            assert isinstance(sample_dict[f], (int, float))
