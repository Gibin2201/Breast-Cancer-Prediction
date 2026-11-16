"""
Unit Tests for Tumor Prediction and Inference Pipeline.
Verifies input validation, numerical bounds, probability outputs, and clinical disclaimers.
"""

import pytest
from pathlib import Path

from config import BEST_MODEL_PATH, FEATURE_NAMES
from utils.prediction import predict_tumor
from utils.data_loader import load_dataset, get_sample_records
from train_model import train_and_evaluate_all


@pytest.fixture(scope="module")
def sample_data():
    """Ensure model is trained and provide sample records."""
    if not BEST_MODEL_PATH.exists():
        train_and_evaluate_all()
    df = load_dataset()
    samples = get_sample_records(df)
    return samples


def test_benign_prediction(sample_data):
    """Verify prediction pipeline on a known Benign sample."""
    benign_sample = list(sample_data.values())[0]
    result = predict_tumor(benign_sample)

    assert result["prediction_code"] in [0, 1]
    assert result["prediction_label"] in ["Benign", "Malignant"]
    assert 0.0 <= result["benign_probability"] <= 1.0
    assert 0.0 <= result["malignant_probability"] <= 1.0
    assert pytest.approx(result["benign_probability"] + result["malignant_probability"], rel=1e-2) == 1.0
    assert "clinical_explanation" in result
    assert "disclaimer" in result
    assert "Educational" in result["disclaimer"]
    assert "model_name" in result
    assert 0.0 <= result["model_accuracy"] <= 1.0


def test_malignant_prediction(sample_data):
    """Verify prediction pipeline on a known Malignant sample."""
    malignant_sample = list(sample_data.values())[1]
    result = predict_tumor(malignant_sample)

    assert result["prediction_code"] in [0, 1]
    assert result["prediction_label"] in ["Benign", "Malignant"]
    assert 0.0 <= result["dominant_confidence"] <= 100.0


def test_input_validation_missing_feature(sample_data):
    """Verify that omitting a feature triggers a ValueError."""
    sample = list(sample_data.values())[0].copy()
    del sample[FEATURE_NAMES[0]]

    with pytest.raises(ValueError, match="Expected exactly 30 features"):
        predict_tumor(sample)


def test_input_validation_negative_value(sample_data):
    """Verify that negative anatomical measurements trigger a ValueError."""
    sample = list(sample_data.values())[0].copy()
    sample[FEATURE_NAMES[0]] = -5.0

    with pytest.raises(ValueError, match="cannot be negative"):
        predict_tumor(sample)


def test_input_validation_non_numeric(sample_data):
    """Verify that non-numeric values trigger a ValueError."""
    sample = list(sample_data.values())[0].copy()
    sample[FEATURE_NAMES[0]] = "invalid_string"

    with pytest.raises(ValueError, match="non-numeric"):
        predict_tumor(sample)
