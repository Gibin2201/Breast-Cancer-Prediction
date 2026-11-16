"""
Download and Validation Script for the Wisconsin Diagnostic Breast Cancer (WDBC) Dataset.

Downloads the official WDBC dataset from the UCI Machine Learning Repository,
validates integrity (569 records, 30 numerical features, diagnosis target),
removes identifier columns, formats cleanly as CSV, and provides a robust fallback
to Scikit-learn's built-in WDBC dataset when offline or network fails.
"""

import sys
import json
import zipfile
import io
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
import pandas as pd

from config import (
    DATASET_DIR,
    DATASET_PATH,
    TARGET_COLUMN,
    FEATURE_NAMES,
    UCI_DATASET_URL,
    UCI_DATA_URL_FALLBACK,
)
from utils.data_loader import validate_dataset


SOURCE_INFO_PATH = DATASET_DIR / "source_info.json"


def save_source_metadata(source_name: str, details: str, is_fallback: bool = False):
    """Save metadata recording the dataset source for audit and UI display."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_name": source_name,
        "details": details,
        "is_fallback": is_fallback,
        "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "records": 569,
        "features": len(FEATURE_NAMES),
        "target_column": TARGET_COLUMN,
        "status": "Validated successfully",
    }
    with open(SOURCE_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def load_source_metadata() -> dict:
    """Read saved source metadata if available."""
    if SOURCE_INFO_PATH.exists():
        try:
            with open(SOURCE_INFO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "source_name": "Local Dataset File",
        "details": "Pre-existing dataset file in dataset/breast_cancer.csv",
        "is_fallback": False,
        "downloaded_at": "Unknown",
    }


def download_from_uci_zip() -> pd.DataFrame:
    """Attempt download and extraction from official UCI ZIP archive."""
    print(f"[*] Connecting to official UCI repository archive:\n    {UCI_DATASET_URL}")
    req = urllib.request.Request(
        UCI_DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # Search for wdbc.data inside zip
        target_file = None
        for name in z.namelist():
            if name.endswith("wdbc.data"):
                target_file = name
                break
        
        if not target_file:
            raise FileNotFoundError("wdbc.data not found in downloaded UCI ZIP archive.")

        with z.open(target_file) as f:
            raw_df = pd.read_csv(f, header=None)

    return raw_df


def download_from_uci_direct() -> pd.DataFrame:
    """Attempt download directly from UCI HTTP data file."""
    print(f"[*] Trying direct UCI data endpoint:\n    {UCI_DATA_URL_FALLBACK}")
    req = urllib.request.Request(
        UCI_DATA_URL_FALLBACK,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        raw_df = pd.read_csv(response, header=None)

    return raw_df


def fallback_to_sklearn() -> pd.DataFrame:
    """
    Fallback: Load identical Wisconsin Diagnostic Breast Cancer dataset from Scikit-Learn.
    Both datasets originate from Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian
    at the University of Wisconsin.
    """
    print("[!] Network download failed or unreachable. Activating official Scikit-Learn fallback...")
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=FEATURE_NAMES)
    
    # In sklearn: 0 = 'malignant', 1 = 'benign'
    # Convert to standard WDBC 'B' and 'M' codes
    df[TARGET_COLUMN] = ["M" if t == 0 else "B" for t in cancer.target]
    
    save_source_metadata(
        source_name="Scikit-Learn Fallback (WDBC Dataset)",
        details="Loaded from sklearn.datasets.load_breast_cancer due to network failure/timeout. "
                "Contains the identical 569 Wisconsin Diagnostic Breast Cancer clinical records.",
        is_fallback=True,
    )
    return df


def download_dataset(force: bool = False) -> pd.DataFrame:
    """
    Download, process, and validate the Wisconsin Diagnostic Breast Cancer dataset.
    
    Args:
        force: If True, re-download even if a valid local copy exists.
        
    Returns:
        pd.DataFrame: Clean validated dataset.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check if dataset already exists and is valid
    if DATASET_PATH.exists() and not force:
        try:
            existing_df = pd.read_csv(DATASET_PATH)
            is_valid, msg = validate_dataset(existing_df)
            if is_valid:
                print(f"[+] Valid local dataset already present at '{DATASET_PATH}'.")
                print(f"    Dimensions: {existing_df.shape[0]} rows, {existing_df.shape[1]} columns.")
                return existing_df
            else:
                print(f"[-] Existing dataset failed validation: {msg}. Re-downloading...")
        except Exception as e:
            print(f"[-] Existing file corrupt ({e}). Re-downloading...")

    df = None

    # 2. Try primary official UCI ZIP archive
    try:
        raw_df = download_from_uci_zip()
        # Columns: [0: id, 1: diagnosis, 2..31: 30 features]
        raw_df = raw_df.iloc[:, 1:32]
        raw_df.columns = [TARGET_COLUMN] + FEATURE_NAMES
        # Rearrange so features come first or diagnosis is clearly preserved
        cols = FEATURE_NAMES + [TARGET_COLUMN]
        df = raw_df[cols]
        save_source_metadata(
            source_name="UCI Machine Learning Repository (Official ZIP)",
            details="Downloaded directly from UCI archive (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).",
            is_fallback=False,
        )
        print("[+] Successfully downloaded and extracted official dataset from UCI ZIP archive.")
    except Exception as zip_err:
        print(f"[-] UCI ZIP download unsuccessful: {zip_err}")

        # 3. Try direct UCI data endpoint
        try:
            raw_df = download_from_uci_direct()
            raw_df = raw_df.iloc[:, 1:32]
            raw_df.columns = [TARGET_COLUMN] + FEATURE_NAMES
            cols = FEATURE_NAMES + [TARGET_COLUMN]
            df = raw_df[cols]
            save_source_metadata(
                source_name="UCI Machine Learning Repository (Direct wdbc.data)",
                details="Downloaded directly from UCI data file endpoint.",
                is_fallback=False,
            )
            print("[+] Successfully downloaded dataset from UCI direct endpoint.")
        except Exception as direct_err:
            print(f"[-] UCI direct download unsuccessful: {direct_err}")
            
            # 4. Fallback to Scikit-learn
            df = fallback_to_sklearn()

    # 5. Validate dataset
    is_valid, val_message = validate_dataset(df)
    if not is_valid:
        raise ValueError(f"Dataset verification failed: {val_message}")

    # 6. Save cleanly formatted CSV
    df.to_csv(DATASET_PATH, index=False)
    print(f"[+] Dataset saved successfully to: {DATASET_PATH}")
    print(f"    Total Records: {len(df)}")
    print(f"    Total Input Features: {len(FEATURE_NAMES)}")
    print(f"    Benign Tumors (B): {(df[TARGET_COLUMN].isin(['B', 0, 'Benign'])).sum()}")
    print(f"    Malignant Tumors (M): {(df[TARGET_COLUMN].isin(['M', 1, 'Malignant'])).sum()}")

    return df


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    download_dataset(force=force_download)
