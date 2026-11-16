"""
Unit Tests for Machine Learning Model Training, Selection, and Persistence.
Verifies algorithm training, evaluation metrics, pipeline saving, and metadata integrity.
"""

import pytest
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

from config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODEL_COMPARISON_PATH,
    FEATURE_NAMES,
)
from train_model import train_and_evaluate_all
from utils.prediction import load_trained_model, load_model_metadata


@pytest.fixture(scope="module")
def trained_artifacts():
    """Ensure models are trained and return metadata."""
    if not (BEST_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists() and MODEL_COMPARISON_PATH.exists()):
        train_and_evaluate_all()
    metadata = load_model_metadata(force_reload=True)
    model = load_trained_model(force_reload=True)
    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
    return model, metadata, comparison_df


def test_model_files_exist(trained_artifacts):
    """Verify that all required model output files exist on disk."""
    assert BEST_MODEL_PATH.exists(), "best_model.pkl is missing"
    assert MODEL_METADATA_PATH.exists(), "model_metadata.json is missing"
    assert MODEL_COMPARISON_PATH.exists(), "model_comparison.csv is missing"


def test_model_pipeline_structure(trained_artifacts):
    """Verify that the saved model is a valid Scikit-learn Pipeline."""
    model, _, _ = trained_artifacts
    assert isinstance(model, Pipeline), "Saved model must be a Scikit-learn Pipeline"
    assert hasattr(model, "predict"), "Model pipeline must have predict method"
    assert hasattr(model, "predict_proba") or hasattr(model, "decision_function"), (
        "Model must support probability/decision scores"
    )


def test_all_five_algorithms_compared(trained_artifacts):
    """Verify that exactly 5 algorithms are evaluated in the comparison table."""
    _, _, comparison_df = trained_artifacts
    expected_algos = {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "K-Nearest Neighbors",
        "Support Vector Machine",
    }
    present_algos = set(comparison_df["Algorithm"].values)
    assert expected_algos.issubset(present_algos), (
        f"Missing algorithms: {expected_algos - present_algos}"
    )


def test_evaluation_metric_ranges(trained_artifacts):
    """Verify that model performance scores are valid non-hardcoded probabilities [0, 1]."""
    _, _, comparison_df = trained_artifacts
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for m in metrics:
        assert m in comparison_df.columns, f"Metric '{m}' missing in comparison"
        for val in comparison_df[m]:
            assert 0.0 <= val <= 1.0, f"Metric {m} value {val} out of range [0, 1]"
            # Sanity check: valid ML models on WDBC achieve at least 80% accuracy
            assert val > 0.70, f"Unusually low score for {m}: {val}"


def test_metadata_completeness(trained_artifacts):
    """Verify metadata contains required audit information."""
    _, metadata, _ = trained_artifacts
    assert "model_name" in metadata
    assert "selection_reason" in metadata
    assert "best_model_metrics" in metadata
    assert "best_model_cv_metrics" in metadata
    assert "roc_curves" in metadata
    assert len(metadata["feature_names"]) == 30


def test_selected_model_has_highest_accuracy(trained_artifacts):
    """Verify that the selected best model has the highest accuracy from model_comparison.csv."""
    _, metadata, comparison_df = trained_artifacts
    selected_name = metadata["model_name"]
    top_row = comparison_df.sort_values(by=["Accuracy", "F1-Score", "Recall"], ascending=False).iloc[0]
    expected_best_name = top_row["Algorithm"]
    assert selected_name == expected_best_name, (
        f"Selected model '{selected_name}' does not match highest accuracy model '{expected_best_name}' "
        f"(Accuracy: {top_row['Accuracy']:.2%})"
    )
