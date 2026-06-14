# src/preprocess.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stage 0 â€” Data Preprocessing
#
# What this file does:
#   1. Loads the Pima Indians Diabetes CSV
#   2. Replaces biologically impossible zero values with column medians
#   3. Normalizes all features to [0, 1] using MinMaxScaler
#   4. Splits data into 80% train / 20% test
#   5. Returns everything needed by the three model stages
#
# Why we replace zeros:
#   Columns like Glucose, BloodPressure, BMI cannot be 0 in real life.
#   These are actually missing values coded as 0 in this dataset.
#   Replacing with median is a standard imputation technique.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Columns where 0 is biologically impossible â†’ treat as missing
ZERO_NOT_VALID = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

FEATURE_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age'
]
TARGET_COL = 'Outcome'


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and fix invalid zero values using median imputation.
    Returns a cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)

    print(f"[Preprocess] Loaded {len(df)} rows, {df.shape[1]} columns.")

    # Replace 0s with NaN in columns where 0 is impossible
    for col in ZERO_NOT_VALID:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            df[col] = df[col].replace(0, np.nan)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[Preprocess] '{col}': replaced {zero_count} zero(s) with median ({median_val:.2f})")

    print(f"[Preprocess] Cleaning done. No null values: {df.isnull().sum().sum() == 0}")
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Normalize features using MinMaxScaler and split into train/test.

    Returns:
        X_train, X_test, y_train, y_test  (numpy arrays)
        scaler                             (fitted MinMaxScaler â€” saved for use in UI)
        feature_names                      (list of column names)
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Scale features to [0, 1]
    # MinMaxScaler: x_scaled = (x - x_min) / (x_max - x_min)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split â€” preserves class ratio in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[Preprocess] Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"[Preprocess] Class balance in train â€” 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")

    return X_train, X_test, y_train, y_test, scaler, FEATURE_COLS


def save_scaler(scaler, path: str = "models/scaler.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"[Preprocess] Scaler saved to {path}")


def load_scaler(path: str = "models/scaler.pkl"):
    return joblib.load(path)
