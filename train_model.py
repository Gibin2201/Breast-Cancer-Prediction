"""
Model Training and Selection Pipeline for Breast Cancer Prediction.

Trains, evaluates, and cross-validates 5 Machine Learning algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

Performs 5-Fold Stratified Cross-Validation on the training set,
evaluates on a held-out stratified test set (20%), and selects the best
model based on Malignant Recall, F1-Score, and ROC-AUC.
Saves:
- models/best_model.pkl
- models/model_metadata.json
- models/model_comparison.csv
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from config import (
    MODELS_DIR,
    BEST_MODEL_PATH,
    ALL_MODELS_PATH,
    MODEL_METADATA_PATH,
    MODEL_COMPARISON_PATH,
    FEATURE_NAMES,
    RANDOM_SEED,
    TEST_SIZE,
    CV_FOLDS,
    DATASET_PATH,
)
from download_dataset import download_dataset, load_source_metadata
from utils.data_loader import load_dataset
from utils.preprocessing import prepare_features_and_target, split_data, create_pipeline
from utils.model_evaluation import (
    evaluate_single_model,
    perform_cross_validation,
    select_best_model,
)


def get_algorithms(random_state: int = RANDOM_SEED):
    """
    Define 5 candidate classification algorithms with tuned, robust hyperparameters.
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "use_scaler": True,
            "description": "Linear classifier with balanced class weights and L2 regularization.",
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            ),
            "use_scaler": False,
            "description": "Interpretable tree-based non-linear partitioning classifier.",
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=4,
                class_weight="balanced",
                random_state=random_state,
            ),
            "use_scaler": False,
            "description": "Ensemble of 100 decorrelated bagged decision trees.",
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="minkowski",
            ),
            "use_scaler": True,
            "description": "Non-parametric distance-based instance classifier with inverse distance weighting.",
        },
        "Support Vector Machine": {
            "estimator": SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
            "use_scaler": True,
            "description": "Radial Basis Function (RBF) margin maximizer with probability calibration.",
        },
    }


def train_and_evaluate_all(df: pd.DataFrame = None):
    """
    Train and rigorously evaluate all 5 algorithms.
    """
    # 1. Ensure dataset exists
    if df is None:
        if not DATASET_PATH.exists():
            print("[*] Dataset not found. Downloading...")
            df = download_dataset()
        else:
            df = load_dataset()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Extract features and target
    X, y = prepare_features_and_target(df)
    print(f"[+] Loaded {len(df)} samples with {X.shape[1]} features.")
    print(f"    Target distribution -> Benign: {(y == 0).sum()}, Malignant: {(y == 1).sum()}")

    # 3. Stratified Train / Test split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    print(f"[+] Stratified Split: {len(X_train)} Train samples, {len(X_test)} Test samples (80/20).")

    algorithms = get_algorithms()
    fitted_pipelines = {}
    evaluation_results = {}
    comparison_rows = []
    roc_curves_data = {}

    print("\n" + "=" * 70)
    print(" TRAINING & EVALUATING 5 MACHINE LEARNING ALGORITHMS ")
    print("=" * 70)

    for name, config in algorithms.items():
        print(f"\n--> Training: {name} ...")
        pipeline = create_pipeline(classifier=config["estimator"], use_scaler=config["use_scaler"])

        # Cross-validation on training data to prevent data leakage and test-set selection bias
        cv_metrics = perform_cross_validation(
            pipeline, X_train, y_train, cv_folds=CV_FOLDS, random_state=RANDOM_SEED
        )

        # Fit final pipeline on full training split
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on held-out test split
        test_metrics = evaluate_single_model(pipeline, X_test, y_test)
        evaluation_results[name] = {
            "test_metrics": test_metrics,
            "cv_metrics": cv_metrics,
            "description": config["description"],
            "params": {k: str(v) for k, v in config["estimator"].get_params().items() if k in ["C", "kernel", "n_estimators", "max_depth", "n_neighbors", "weights", "class_weight"]},
        }

        roc_curves_data[name] = {
            "fpr": test_metrics["roc_curve"]["fpr"],
            "tpr": test_metrics["roc_curve"]["tpr"],
            "auc": test_metrics["roc_auc"],
        }

        # Tabular summary row
        comparison_rows.append({
            "Algorithm": name,
            "Accuracy": test_metrics["accuracy"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "F1-Score": test_metrics["f1_score"],
            "ROC-AUC": test_metrics["roc_auc"],
            "CV Accuracy": cv_metrics["cv_accuracy_mean"],
            "CV Precision": cv_metrics["cv_precision_mean"],
            "CV Recall": cv_metrics["cv_recall_mean"],
            "CV F1": cv_metrics["cv_f1_mean"],
            "CV ROC-AUC": cv_metrics["cv_roc_auc_mean"],
        })

        print(f"    Test Accuracy : {test_metrics['accuracy']:.2%}")
        print(f"    Test Recall   : {test_metrics['recall']:.2%} (Malignant sensitivity)")
        print(f"    Test F1-Score : {test_metrics['f1_score']:.2%}")
        print(f"    Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
        print(f"    CV F1 (Mean)  : {cv_metrics['cv_f1_mean']:.2%} (+/- {cv_metrics['cv_f1_std']:.2%})")

    # 4. Save model comparison CSV sorted by Accuracy descending
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values(by=["Accuracy", "F1-Score", "Recall"], ascending=False)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\n[+] Model comparison table saved to: {MODEL_COMPARISON_PATH}")

    # 5. Automatically select best model with highest test accuracy
    best_name, selection_reason = select_best_model(comparison_rows)
    best_pipeline = fitted_pipelines[best_name]
    best_eval = evaluation_results[best_name]

    print("\n" + "=" * 70)
    print(f" OPTIMAL MODEL SELECTED (HIGHEST ACCURACY): {best_name.upper()} ")
    print("=" * 70)
    print(f"Reason: {selection_reason}\n")

    # 6. Save best model pipeline
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"[+] Saved best model pipeline to: {BEST_MODEL_PATH}")

    # Also persist all fitted pipelines for multi-model options
    joblib.dump(fitted_pipelines, ALL_MODELS_PATH)
    print(f"[+] Saved all algorithm pipelines to: {ALL_MODELS_PATH}")

    # 7. Save complete metadata JSON
    dataset_source = load_source_metadata()
    metadata = {
        "model_name": best_name,
        "selection_reason": selection_reason,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_info": {
            "source": dataset_source.get("source_name", "UCI WDBC"),
            "details": dataset_source.get("details", ""),
            "total_records": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(FEATURE_NAMES),
            "target_classes": ["Benign", "Malignant"],
        },
        "feature_names": FEATURE_NAMES,
        "best_model_metrics": best_eval["test_metrics"],
        "best_model_cv_metrics": best_eval["cv_metrics"],
        "best_model_hyperparameters": best_eval["params"],
        "all_algorithms_evaluated": list(algorithms.keys()),
        "roc_curves": roc_curves_data,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved comprehensive model metadata to: {MODEL_METADATA_PATH}")

    return {
        "best_model_name": best_name,
        "selection_reason": selection_reason,
        "comparison_df": comparison_df,
        "metadata": metadata,
    }


if __name__ == "__main__":
    train_and_evaluate_all()
