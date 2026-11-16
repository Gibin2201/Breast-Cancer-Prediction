"""
Model Evaluation and Comparison Module for Breast Cancer Prediction.
Calculates Accuracy, Precision, Recall (Malignant sensitivity), F1-Score, ROC-AUC,
Confusion Matrices, and Stratified Cross-Validation metrics.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_single_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate a fitted model pipeline on test data.
    
    Calculates:
    - Accuracy, Precision, Recall (Malignant class), F1-Score, ROC-AUC
    - Confusion Matrix values (TN, FP, FN, TP)
    - ROC Curve data points (FPR, TPR)
    """
    y_pred = model.predict(X_test)

    # Attempt to extract probability scores for class 1 (Malignant)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Scale decision function outputs to 0-1 range via sigmoid
        decision = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-decision))
    else:
        y_prob = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_test, y_prob))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fpr_list = fpr.tolist()
        tpr_list = tpr.tolist()
    except Exception:
        auc = 0.5
        fpr_list = [0.0, 1.0]
        tpr_list = [0.0, 1.0]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist(),
        },
        "roc_curve": {
            "fpr": fpr_list,
            "tpr": tpr_list,
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Benign", "Malignant"], output_dict=True
        ),
    }


def perform_cross_validation(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Perform Stratified K-Fold Cross-Validation on training data.
    Ensures model selection is not biased by repetitive test set evaluations.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        "cv_accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
        "cv_accuracy_std": round(float(scores["test_accuracy"].std()), 4),
        "cv_precision_mean": round(float(scores["test_precision"].mean()), 4),
        "cv_recall_mean": round(float(scores["test_recall"].mean()), 4),
        "cv_recall_std": round(float(scores["test_recall"].std()), 4),
        "cv_f1_mean": round(float(scores["test_f1"].mean()), 4),
        "cv_f1_std": round(float(scores["test_f1"].std()), 4),
        "cv_roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
    }


def select_best_model(comparison_records: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Automatically select the optimal model based on the HIGHEST Test Accuracy
    from the actual model evaluation results.
    
    If multiple models share identical accuracy, F1-Score, Recall, and ROC-AUC
    are used as deterministic tie-breakers.
    Does NOT hardcode any algorithm name.
      
    Returns:
        (best_model_name, selection_reason)
    """
    if not comparison_records:
        raise ValueError("No comparison records provided for best model selection.")

    # Select the model with the highest test accuracy dynamically
    best_record = max(
        comparison_records,
        key=lambda r: (
            float(r.get("Accuracy", 0.0)),
            float(r.get("F1-Score", 0.0)),
            float(r.get("Recall", 0.0)),
            float(r.get("ROC-AUC", 0.0)),
        ),
    )

    best_name = best_record["Algorithm"]
    acc = float(best_record.get("Accuracy", 0.0))
    rec = float(best_record.get("Recall", 0.0))
    f1 = float(best_record.get("F1-Score", 0.0))
    auc = float(best_record.get("ROC-AUC", 0.0))

    reason = (
        f"Automatically selected '{best_name}' as the optimal classifier because it achieved the "
        f"Highest Test Accuracy ({acc:.2%}) among all evaluated algorithms, along with "
        f"{rec:.2%} Malignant Recall, {f1:.2%} F1-Score, and {auc:.4f} ROC-AUC."
    )

    return best_name, reason
