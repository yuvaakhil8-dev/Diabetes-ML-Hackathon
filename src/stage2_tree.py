# src/stage2_tree.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stage 2 â€” Diagnosis with Reasoning via Decision Tree
#
# Role in the pipeline:
#   Receives only HIGH RISK patients flagged by Stage 1.
#   Classifies each as diabetic (1) or non-diabetic (0).
#   Also explains WHICH features drove the decision (information gain).
#
# Two variants trained:
#   - Gini impurity  : measures probability of misclassification
#   - Entropy        : measures information disorder (from information theory)
#   Both are compared. The better one is selected as the final model.
#
# Information Gain:
#   IG(feature) = Entropy(parent) - weighted average Entropy(children)
#   Higher IG = feature reduces uncertainty more = better split point
#   We compute and display IG for every feature as a bar chart.
#
# Metrics reported:
#   Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
#   + Gini vs Entropy comparison table
#   + Information Gain bar chart data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def compute_entropy(y: np.ndarray) -> float:
    """
    Shannon entropy of a label array.
    H(S) = -sum( p_i * log2(p_i) )
    Returns 0 if only one class present.
    """
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    # Avoid log(0) by filtering zero probabilities
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_information_gain(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """
    Compute Information Gain for each feature using a binary split at the median.

    IG(feature) = H(S) - [ (|S_left|/|S|) * H(S_left) + (|S_right|/|S|) * H(S_right) ]

    This is a simplified version of what the Decision Tree does internally
    at each node to choose the best split.
    """
    parent_entropy = compute_entropy(y)
    ig_scores = {}

    for i, feature in enumerate(feature_names):
        col = X[:, i]
        median_val = np.median(col)

        # Split on median
        left_mask  = col <= median_val
        right_mask = col >  median_val

        y_left  = y[left_mask]
        y_right = y[right_mask]

        n = len(y)
        n_left  = len(y_left)
        n_right = len(y_right)

        # Weighted entropy of children
        if n_left == 0 or n_right == 0:
            ig = 0.0
        else:
            weighted_child_entropy = (
                (n_left  / n) * compute_entropy(y_left) +
                (n_right / n) * compute_entropy(y_right)
            )
            ig = parent_entropy - weighted_child_entropy

        ig_scores[feature] = round(ig, 4)

    # Sort descending
    ig_scores = dict(sorted(ig_scores.items(), key=lambda x: x[1], reverse=True))
    return ig_scores


def train(X_train: np.ndarray, y_train: np.ndarray, criterion: str = "gini",
          max_depth: int = 5, random_state: int = 42) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.

    criterion: 'gini' or 'entropy'
    max_depth:  limits tree depth to prevent overfitting.
                Shallow trees are also easier to explain to the panel.
    """
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print(f"[Stage 2] Decision Tree trained with criterion='{criterion}', max_depth={max_depth}")
    return model


def evaluate(model: DecisionTreeClassifier, X_test: np.ndarray,
             y_test: np.ndarray, label: str = "Decision Tree") -> dict:
    """
    Compute all classification metrics for the Decision Tree.
    Returns a dict with all metrics + raw predictions.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"[Stage 2] {label} â†’ Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

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


def get_feature_importances(model: DecisionTreeClassifier, feature_names: list) -> dict:
    """
    Returns sklearn's built-in feature importance scores.
    These are based on the total reduction in impurity (Gini/Entropy)
    contributed by each feature across all splits in the tree.
    Higher = more important.
    """
    importances = model.feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def compare_criteria(X_train, y_train, X_test, y_test) -> tuple:
    """
    Train one tree with Gini and one with Entropy.
    Evaluate both and return results + the better model.
    """
    model_gini    = train(X_train, y_train, criterion="gini")
    model_entropy = train(X_train, y_train, criterion="entropy")

    results_gini    = evaluate(model_gini,    X_test, y_test, label="Decision Tree (Gini)")
    results_entropy = evaluate(model_entropy, X_test, y_test, label="Decision Tree (Entropy)")

    # Pick the one with higher F1 score as the final model
    if results_gini["f1"] >= results_entropy["f1"]:
        best_model   = model_gini
        best_results = results_gini
        best_label   = "Gini"
    else:
        best_model   = model_entropy
        best_results = results_entropy
        best_label   = "Entropy"

    print(f"[Stage 2] Best criterion: {best_label}")
    return best_model, best_results, results_gini, results_entropy


def save_model(model: DecisionTreeClassifier, path: str = "models/stage2_tree.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Stage 2] Model saved to {path}")


def load_model(path: str = "models/stage2_tree.pkl") -> DecisionTreeClassifier:
    return joblib.load(path)
