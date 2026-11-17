---
title: Breast Cancer Prediction AI Platform
emoji: 🎗️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
license: mit
---

# Breast Cancer Prediction Using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Framework](https://img.shields.io/badge/Web%20UI-Streamlit-red.svg)](https://streamlit.io/)
[![Testing](https://img.shields.io/badge/Testing-Pytest-yellow.svg)](https://docs.pytest.org/)
[![Academic Level](https://img.shields.io/badge/Project-MCA%20Micro%20Project-teal.svg)](#)

A complete, production-grade academic Machine Learning clinical decision support web application developed for **Master of Computer Applications (MCA)** micro project demonstration. The system classifies breast tumors into **Benign (Non-cancerous)** or **Malignant (Cancerous)** based on 30 digitized fine needle aspirate (FNA) cell nuclei morphometric measurements from the benchmark **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

---

## 📌 1. Project Overview

Breast cancer is the most prevalent malignancy among women globally. Early and reliable diagnosis significantly improves clinical prognosis, increasing five-year survival rates to over 99% for localized carcinomas.

This application provides a comprehensive end-to-end data science and machine learning workflow:
- **Automated Dataset Ingestion:** Downloads and validates the official UCI WDBC dataset with automatic offline Scikit-Learn fallback.
- **Exploratory Data Analysis (EDA):** Interactive Plotly distributions, correlation heatmaps, box plots, and bi-variate scatter clusters.
- **Leakage-Free Preprocessing:** Standardized Scikit-learn `Pipeline` architectures that strictly isolate training transformations from test evaluation and live inference.
- **Multi-Algorithm Tournament:** Benchmarks 5 distinct machine learning algorithms with 5-fold stratified cross-validation:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)
- **Clinical Model Selection:** Objective selection prioritizing **Malignant Recall (Sensitivity)** to protect against critical false negatives (missed carcinomas).
- **Interactive Web Application:** Built with Streamlit, featuring a modern healthcare aesthetic (Navy, Teal, Blue), preset real patient cases, 30-feature validated inputs, and calibrated probability score gauges.

---

## 🏗️ 2. Project Architecture & Folder Structure

```
Breast-Cancer-Prediction/
│
├── app.py                      # Main Streamlit application entrypoint & routing
├── download_dataset.py         # Automated UCI downloader & integrity validator
├── train_model.py              # ML training, cross-validation & model selection pipeline
├── requirements.txt            # Complete production Python dependency manifest
├── README.md                   # Comprehensive academic documentation
├── .gitignore                  # Git repository exclusion rules
├── config.py                   # Central configuration, paths, features & theme palette
│
├── dataset/
│   ├── breast_cancer.csv       # Standardized 569 x 31 dataset (Features + Target)
│   └── source_info.json        # Audit trail recording dataset acquisition source
│
├── models/
│   ├── best_model.pkl          # Saved optimal Scikit-learn pipeline (Joblib)
│   ├── model_comparison.csv    # Benchmark comparison metrics across 5 algorithms
│   └── model_metadata.json     # Hyperparameters, test metrics & ROC curve coordinates
│
├── pages/
│   ├── 1_Dashboard.py          # Clinical telemetry, class distributions & model status
│   ├── 2_Dataset_Overview.py   # Dataset specifications, df.head(5), df.tail(5), stats
│   ├── 3_Dataset_Analysis.py   # Exploratory Data Analysis (EDA), heatmaps, boxplots
│   ├── 4_Data_Preprocessing.py # Interactive 6-step preprocessing pipeline & scaling
│   ├── 5_Machine_Learning_Models.py # 5 algorithm profiles, formulas & hyperparameters
│   ├── 6_Model_Comparison.py   # Benchmark leaderboard, grouped bar chart, radar chart
│   ├── 7_Performance_Evaluation.py # Confusion matrices, multi-ROC curves, clinical recall
│   ├── 8_Prediction.py         # 30-feature real-time inference form & risk gauge
│   └── 9_About_Project.py      # Academic defense brief, mathematical formulations
│
├── utils/
│   ├── __init__.py             # Python utility package descriptor
│   ├── data_loader.py          # CSV loading, validation & patient preset extractors
│   ├── preprocessing.py       # Pipelines, label encoding & input tensor validators
│   ├── visualization.py        # Modern Plotly chart generators (Heatmaps, ROC, Gauges)
│   ├── model_evaluation.py     # Cross-validation scorers, confusion matrices & selection
│   ├── prediction.py           # Cached pipeline loader & real-time inference engine
│   └── ui_components.py        # Custom healthcare CSS, banners, cards & setup guide
│
├── assets/
│   └── logo.png                # High-resolution healthcare emblem (Pillow generated)
│
└── tests/
    ├── test_dataset.py         # Automated tests for dataset shape, features & nulls
    ├── test_model.py           # Automated tests for model training, metrics & pipeline
    └── test_prediction.py      # Automated tests for validation, bounds & probabilities
```

---

## 🔬 3. Dataset Specifications

- **Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Primary Source:** UCI Machine Learning Repository (ID: 17)
- **Clinical Authors:** Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin Hospitals)
- **Total Instances:** 569 patient biopsies
- **Target Variable:** `diagnosis`
  - `Benign (B / 0)`: 357 cases (62.74%)
  - `Malignant (M / 1)`: 212 cases (37.26%)
- **Number of Input Features:** 30 continuous real-valued nuclear measurements computed from digitized FNA images:

| Group | Features (1 - 10 per group) |
| :--- | :--- |
| **Mean Features (1 - 10)** | Mean Radius, Mean Texture, Mean Perimeter, Mean Area, Mean Smoothness, Mean Compactness, Mean Concavity, Mean Concave Points, Mean Symmetry, Mean Fractal Dimension |
| **Standard Error (11 - 20)** | Radius SE, Texture SE, Perimeter SE, Area SE, Smoothness SE, Compactness SE, Concavity SE, Concave Points SE, Symmetry SE, Fractal Dimension SE |
| **Worst / Extreme (21 - 30)** | Worst Radius, Worst Texture, Worst Perimeter, Worst Area, Worst Smoothness, Worst Compactness, Worst Concavity, Worst Concave Points, Worst Symmetry, Worst Fractal Dimension |

---

## 🤖 4. Machine Learning Algorithms Implemented

| Algorithm | Paradigm | Scaling Required | Core Hyperparameters |
| :--- | :--- | :---: | :--- |
| **Logistic Regression** | Generalized Linear Model | Yes (StandardScaler) | $C=1.0$, `max_iter=1000`, `class_weight='balanced'` |
| **Decision Tree** | Non-parametric Hierarchical | No | `criterion='gini'`, `max_depth=5`, `min_samples_split=5` |
| **Random Forest** | Bagging Ensemble | No | `n_estimators=100`, `max_depth=8`, `min_samples_split=4` |
| **K-Nearest Neighbors** | Instance-Based Metric | Yes (StandardScaler) | $k=5$, `weights='distance'`, metric: Minkowski ($p=2$) |
| **Support Vector Machine** | Maximum Margin Kernel | Yes (StandardScaler) | Kernel: RBF, $C=1.0$, `probability=True`, `class_weight='balanced'` |

### Model Selection Criterion
In clinical tumor triage, **False Negatives (Type II error)** must be strictly minimized. Therefore, model selection avoids naive test accuracy and evaluates candidate algorithms through 5-Fold Stratified Cross-Validation using a weighted composite safety score:
$$\text{Score} = (0.45 \times \text{CV Recall}) + (0.35 \times \text{CV F1-Score}) + (0.20 \times \text{CV ROC-AUC})$$

---

## 💻 5. Installation & Execution Guide

### Prerequisites
- Python 3.11 or Python 3.12
- Windows PowerShell, macOS Terminal, or Linux Bash

### Step 1: Clone or Navigate to the Workspace
```powershell
cd "d:\Breast Cancer Prediction"
```

### Step 2: Create a Dedicated Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
On Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(On macOS / Linux: `source venv/bin/activate`)*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Automatically Download and Validate Dataset
```powershell
python download_dataset.py
```
*Note: If the UCI online server is unreachable, the script automatically activates the official Scikit-Learn WDBC fallback, converting it into identical standardized format.*

### Step 6: Train and Benchmark All 5 Machine Learning Models
```powershell
python train_model.py
```

### Step 7: Launch the Interactive Streamlit Web Application
```powershell
streamlit run app.py
```
The application will launch automatically at: `http://localhost:8501`

---

## 🧪 6. Automated Testing Suite

Comprehensive unit tests verify data integrity, pipeline isolation, model persistence, and input boundary validation.

Execute all automated tests via Pytest:
```powershell
pytest tests/ -v
```

### Test Coverage Summary:
- `test_dataset.py`: Verifies 569 rows, 30 features, presence of target, zero nulls, and sample generation.
- `test_model.py`: Verifies `best_model.pkl`, `model_metadata.json`, evaluation metric bounds, and 5-algorithm comparison.
- `test_prediction.py`: Verifies real-time inference on known samples, negative value rejection, and probability sum constraints.

---

## 📱 7. Application Page Walkthrough

### 1. Clinical Dashboard (`pages/1_Dashboard.py`)
- Project mission statement and clinical scope.
- Top KPI metric cards: Total Biopsies, Features, Benign Count, Malignant Count.
- Interactive Plotly diagnosis donut chart.
- Active algorithm telemetry with live test accuracy, malignant recall, F1, and ROC-AUC.
- Quick preview distributions of nuclear radius and concavity.

### 2. Dataset Analysis / EDA (`pages/2_Dataset_Analysis.py`)
- Filterable clinical dataset table with one-click CSV export.
- Complete descriptive statistics table across all 30 attributes.
- Missing and duplicate value integrity audit (100% complete).
- Interactive Pearson correlation heatmap grouped by feature cohorts.
- Comparative distribution charts (Mean Radius, Texture, Area) separated by diagnosis.
- Multi-feature box plots and 2D bi-variate scatter clustering.

### 3. Multi-Model Comparison (`pages/3_Model_Comparison.py`)
- Tournament winner announcement with clinical decision justification.
- Tabular benchmark comparing all 5 algorithms on held-out test splits and cross-validation.
- Multi-metric grouped bar charts.
- Confusion Matrix heatmap with Type I and Type II error annotations.
- Multi-algorithm Receiver Operating Characteristic (ROC) curves with random chance baseline.
- Interactive "Retrain All 5 Algorithms" trigger button.

### 4. Real-Time Tumor Prediction (`pages/4_Breast_Cancer_Prediction.py`)
- Preset buttons to load verified real Benign (Patient #19) and Malignant (Patient #0) biopsy records.
- 30 numeric inputs organized into three expandable tabs (Mean, SE, Worst).
- Built-in boundary validation preventing negative physical dimensions.
- Responsive diagnosis outcome banner:
  - **Benign:** Emerald Green informational card (`#ecfdf5`).
  - **Malignant:** Crimson Red warning card (`#fef2f2`).
- Calibrated probability table and interactive Plotly malignancy gauge meter.
- Prominent academic medical disclaimer.

### 5. About Project & Documentation (`pages/5_About_Project.py`)
- Problem statement, relevance, and clinical objectives.
- Data science lifecycle diagram and data leakage prevention strategy.
- Mathematical formulations for Logistic Regression, Gini Impurity, Random Forests, Minkowski distance, and RBF kernels.
- Technology stack details, project limitations, future enhancement roadmap, and academic citations.

---

## 🛡️ 8. Clinical & Educational Disclaimer

> **Important Notice:** This software is an academic engineering demonstration created for Master of Computer Applications (MCA) curriculum evaluation. The predictions and statistical probability estimates generated by the machine learning models represent computational approximations derived from the Wisconsin Diagnostic Breast Cancer dataset. This software is **not a medical device**, has **not received regulatory clearance (FDA / CE)**, and must **never be utilized as a substitute for professional clinical judgment, pathological biopsy, or medical diagnosis.**

---

## 📚 9. References

1. **Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993).** *Nuclear feature extraction for breast tumor diagnosis.* IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, vol. 1905, pp. 861-870.
2. **Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995).** *Image analysis and machine learning applied to breast cancer diagnosis and prognosis.* Analytical and Quantitative Cytology and Histology, 17(2), 77-87.
3. **UCI Machine Learning Repository:** *Breast Cancer Wisconsin (Diagnostic) Dataset.* [https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
4. **Pedregosa, F. et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
