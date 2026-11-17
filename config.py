"""
Configuration settings for the Breast Cancer Prediction Machine Learning System.
Defines constants, paths, feature names, hyperparameter spaces, and UI styling.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "breast_cancer.csv"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
ALL_MODELS_PATH = MODELS_DIR / "all_models.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Ensure essential directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Specifications
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
)
UCI_DATA_URL_FALLBACK = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
)

TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"

# Diagnosis mappings
LABEL_TO_NUMERIC = {"B": 0, "M": 1}
NUMERIC_TO_LABEL = {0: "Benign", 1: "Malignant"}
CLASS_NAMES = ["Benign", "Malignant"]

# Complete 30 Numerical Features in Exact Standard Order
FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

# Feature Groups for UI and EDA
MEAN_FEATURES = FEATURE_NAMES[0:10]
SE_FEATURES = FEATURE_NAMES[10:20]
WORST_FEATURES = FEATURE_NAMES[20:30]

# Display Name Dictionary for Friendly UI Presentation
FEATURE_DISPLAY_NAMES = {
    "mean radius": "Mean Radius",
    "mean texture": "Mean Texture",
    "mean perimeter": "Mean Perimeter",
    "mean area": "Mean Area",
    "mean smoothness": "Mean Smoothness",
    "mean compactness": "Mean Compactness",
    "mean concavity": "Mean Concavity",
    "mean concave points": "Mean Concave Points",
    "mean symmetry": "Mean Symmetry",
    "mean fractal dimension": "Mean Fractal Dimension",
    "radius error": "Radius SE",
    "texture error": "Texture SE",
    "perimeter error": "Perimeter SE",
    "area error": "Area SE",
    "smoothness error": "Smoothness SE",
    "compactness error": "Compactness SE",
    "concavity error": "Concavity SE",
    "concave points error": "Concave Points SE",
    "symmetry error": "Symmetry SE",
    "fractal dimension error": "Fractal Dimension SE",
    "worst radius": "Worst Radius",
    "worst texture": "Worst Texture",
    "worst perimeter": "Worst Perimeter",
    "worst area": "Worst Area",
    "worst smoothness": "Worst Smoothness",
    "worst compactness": "Worst Compactness",
    "worst concavity": "Worst Concavity",
    "worst concave points": "Worst Concave Points",
    "worst symmetry": "Worst Symmetry",
    "worst fractal dimension": "Worst Fractal Dimension",
}

# Feature Tooltips / Descriptions
FEATURE_TOOLTIPS = {
    "mean radius": "Mean of distances from center to points on perimeter",
    "mean texture": "Standard deviation of gray-scale values",
    "mean perimeter": "Mean size of the core tumor perimeter",
    "mean area": "Mean tumor area in square units",
    "mean smoothness": "Local variation in radius lengths",
    "mean compactness": "Perimeter^2 / area - 1.0 (mean)",
    "mean concavity": "Severity of concave portions of contour (mean)",
    "mean concave points": "Number of concave portions of contour (mean)",
    "mean symmetry": "Nucleus symmetry score (mean)",
    "mean fractal dimension": "Coastline approximation - 1 (mean)",
    "radius error": "Standard error for the mean radius",
    "texture error": "Standard error for gray-scale texture",
    "perimeter error": "Standard error for tumor perimeter",
    "area error": "Standard error for tumor area",
    "smoothness error": "Standard error for local radius variation",
    "compactness error": "Standard error for perimeter^2 / area - 1.0",
    "concavity error": "Standard error for severity of concave portions",
    "concave points error": "Standard error for number of concave portions",
    "symmetry error": "Standard error for nucleus symmetry",
    "fractal dimension error": "Standard error for coastline approximation",
    "worst radius": "Largest ('worst') mean radius value",
    "worst texture": "Largest ('worst') texture standard deviation",
    "worst perimeter": "Largest ('worst') perimeter size",
    "worst area": "Largest ('worst') tumor area",
    "worst smoothness": "Largest ('worst') local radius variation",
    "worst compactness": "Largest ('worst') compactness value",
    "worst concavity": "Largest ('worst') severity of concave portions",
    "worst concave points": "Largest ('worst') number of concave portions",
    "worst symmetry": "Largest ('worst') symmetry score",
    "worst fractal dimension": "Largest ('worst') fractal dimension value",
}

# Machine Learning Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Healthcare SaaS Theme Palettes
# Primary: Deep Navy (#142D4E), Secondary: Medical Blue (#2878C7), Accent: Teal (#16A6A1)
# Background: Light Grey (#F5F7FA), Cards: White (#FFFFFF), Text: Dark Slate (#253247)
THEME_COLORS = {
    "primary": "#142D4E",         # Deep Navy
    "primary_dark": "#0B1D33",
    "secondary": "#2878C7",       # Medical Blue
    "accent": "#16A6A1",          # Teal Accent
    "navy": "#142D4E",
    "navy_light": "#1D3B63",
    "background": "#F5F7FA",      # Light Grey
    "card_bg": "#FFFFFF",         # Pure White Card
    "text": "#253247",            # Dark Slate
    "text_muted": "#5A6E85",      # Subtle Slate
    "border": "#E2E8F0",          # Soft Border
    "benign": "#10B981",          # Clinical Green
    "benign_bg": "#E8F8F5",       # Soft Emerald Tint
    "benign_border": "#A7F3D0",
    "malignant": "#E63946",       # Clinical Warning Red
    "malignant_bg": "#FEE8E8",    # Soft Rose Tint
    "malignant_border": "#FECACA",
}
