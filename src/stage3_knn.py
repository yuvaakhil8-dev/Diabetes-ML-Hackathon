# src/stage3_knn.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stage 3 â€” Validation through Similarity (KNN + Weighted KNN)
#
# Role in the pipeline:
#   After Stage 2 gives a diagnosis, Stage 3 validates it by
#   finding the K most similar patients from the training set
#   and checking what their outcomes were.
#
# KNN (Standard):
#   Each of the K nearest neighbors gets equal voting weight (1/K).
#   Final prediction = majority vote among the K neighbors.
#
# Weighted KNN:
#   Closer neighbors get MORE voting weight (weight = 1/distance).
#   A neighbor at distance 0.1 influences the prediction more than
#   one at distance 0.9. This is often more accurate than standard KNN.
#
# How K is chosen:
#   We train both variants for K = 3, 5, 7, 9.
#   The K with the highest F1 score is selected as the best K.
#   Results across all K values are plotted as a line chart.
#
# Metrics reported:
#   Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
#   + Accuracy vs K line plot data for both variants
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import numpy as np
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


K_VALUES = [3, 5, 7, 9]   # K values to evaluate


def train(X_train: np.ndarray, y_train: np.ndarray,
          k: int = 5, weighted: bool = False) -> KNeighborsClassifier:
    """
    Train a KNN classifier.

    k        : number of neighbors to consider
    weighted : if True â†’ Weighted KNN (weights='distance')
               if False â†’ Standard KNN (weights='uniform')

    Distance metric: Euclidean (default in sklearn)
    Euclidean distance between two points p and q:
        d(p,q) = sqrt( sum( (p_i - q_i)^2 ) )
    """
    weights = "distance" if weighted else "uniform"
    model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric="euclidean")
    model.fit(X_train, y_train)
    label = "Weighted KNN" if weighted else "KNN"
    print(f"[Stage 3] {label} trained with k={k}")
    return model


def evaluate(model: KNeighborsClassifier, X_test: np.ndarray,
             y_test: np.ndarray, label: str = "KNN") -> dict:
    """
    Compute all classification metrics for a KNN model.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"[Stage 3] {label} â†’ Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    return {
        "model_name" : label,
        "accuracy"   : round(acc, 4),
        "precision"  : round(prec, 4),
        "recall"     : round(rec, 4),
        "f1"         : round(f1, 4),
        "roc_auc"    : round(auc, 4),
        "confusion_matrix": cm,
        "y_pred"     : y_pred,
        "y_proba"    : y_proba
    }


def find_best_k(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                k_values: list = K_VALUES) -> dict:
    """
    Evaluate both KNN and Weighted KNN across all K values.
    Returns:
        - k_results: accuracy for each K for both variants (for line plot)
        - best_knn_model, best_knn_results
        - best_wknn_model, best_wknn_results
    """
    k_results = {
        "k_values"      : k_values,
        "knn_accuracy"  : [],
        "wknn_accuracy" : [],
        "knn_f1"        : [],
        "wknn_f1"       : []
    }

    best_knn_f1   = -1
    best_wknn_f1  = -1
    best_knn_model  = None
    best_wknn_model = None
    best_knn_results  = None
    best_wknn_results = None

    for k in k_values:
        # Standard KNN
        knn_model   = train(X_train, y_train, k=k, weighted=False)
        knn_results = evaluate(knn_model, X_test, y_test, label=f"KNN (k={k})")

        # Weighted KNN
        wknn_model   = train(X_train, y_train, k=k, weighted=True)
        wknn_results = evaluate(wknn_model, X_test, y_test, label=f"Weighted KNN (k={k})")

        k_results["knn_accuracy"].append(knn_results["accuracy"])
        k_results["wknn_accuracy"].append(wknn_results["accuracy"])
        k_results["knn_f1"].append(knn_results["f1"])
        k_results["wknn_f1"].append(wknn_results["f1"])

        if knn_results["f1"] > best_knn_f1:
            best_knn_f1      = knn_results["f1"]
            best_knn_model   = knn_model
            best_knn_results = knn_results

        if wknn_results["f1"] > best_wknn_f1:
            best_wknn_f1      = wknn_results["f1"]
            best_wknn_model   = wknn_model
            best_wknn_results = wknn_results

    print(f"[Stage 3] Best KNN       â†’ {best_knn_results['model_name']} | F1: {best_knn_f1:.4f}")
    print(f"[Stage 3] Best Weighted  â†’ {best_wknn_results['model_name']} | F1: {best_wknn_f1:.4f}")

    return {
        "k_results"         : k_results,
        "best_knn_model"    : best_knn_model,
        "best_knn_results"  : best_knn_results,
        "best_wknn_model"   : best_wknn_model,
        "best_wknn_results" : best_wknn_results
    }


def get_neighbors(model: KNeighborsClassifier, X_train: np.ndarray,
                  y_train: np.ndarray, patient_input: np.ndarray,
                  feature_names: list) -> list:
    """
    For a given patient, returns details of their K nearest neighbors
    from the training set. Used in the Streamlit UI to show similar cases.

    Returns a list of dicts: {rank, distance, outcome, feature_values}
    """
    distances, indices = model.kneighbors(patient_input.reshape(1, -1))
    neighbors = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        neighbors.append({
            "rank"          : rank,
            "distance"      : round(float(dist), 4),
            "outcome"       : "Diabetic" if y_train[idx] == 1 else "Non-Diabetic",
            "feature_values": dict(zip(feature_names, X_train[idx]))
        })
    return neighbors


def save_models(best_knn, best_wknn,
                knn_path: str = "models/stage3_knn.pkl",
                wknn_path: str = "models/stage3_wknn.pkl"):
    os.makedirs(os.path.dirname(knn_path), exist_ok=True)
    joblib.dump(best_knn,  knn_path)
    joblib.dump(best_wknn, wknn_path)
    print(f"[Stage 3] Models saved to {knn_path} and {wknn_path}")


def load_models(knn_path: str = "models/stage3_knn.pkl",
                wknn_path: str = "models/stage3_wknn.pkl"):
    return joblib.load(knn_path), joblib.load(wknn_path)
