# src/stage1_regression.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stage 1 â€” Screening via Linear Regression
#
# Role in the pipeline:
#   Every patient gets a continuous risk score between 0 and 1.
#   Score > 0.5  â†’  flagged as HIGH RISK â†’ passed to Stage 2
#   Score <= 0.5 â†’  LOW RISK â†’ exits pipeline here
#
# Why Linear Regression here?
#   Linear regression finds the best-fit line through the training
#   data by minimizing the sum of squared errors. The output is a
#   continuous value (not 0/1), which we interpret as a risk score.
#   This gives us a graded measure of risk rather than a hard cutoff,
#   which is more realistic for a clinical screening context.
#
# Metrics reported:
#   MSE, RMSE, RÂ² (regression metrics, not classification)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


THRESHOLD = 0.5   # risk score above this â†’ HIGH RISK


def train(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Fit a Linear Regression model on training data.
    y_train is binary (0/1) but treated as a continuous target here.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[Stage 1] Linear Regression model trained.")
    return model


def predict_risk_scores(model: LinearRegression, X: np.ndarray) -> np.ndarray:
    """
    Returns raw continuous risk scores for each patient.
    Values are clipped to [0, 1] to keep them interpretable as probabilities.
    """
    scores = model.predict(X)
    scores = np.clip(scores, 0, 1)
    return scores


def apply_threshold(scores: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """
    Convert continuous risk scores to binary labels using threshold.
    Returns: array of 0 (low risk) or 1 (high risk)
    """
    return (scores >= threshold).astype(int)


def get_high_risk_indices(scores: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """
    Returns the indices of patients whose risk score exceeds the threshold.
    These patients are forwarded to Stage 2.
    """
    return np.where(scores >= threshold)[0]


def evaluate(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute regression metrics on the test set.
    Returns a dict with MSE, RMSE, R2 and the predicted scores.
    """
    scores = predict_risk_scores(model, X_test)
    mse  = mean_squared_error(y_test, scores)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, scores)

    print(f"[Stage 1] MSE:  {mse:.4f}")
    print(f"[Stage 1] RMSE: {rmse:.4f}")
    print(f"[Stage 1] RÂ²:   {r2:.4f}")

    return {
        "model_name" : "Linear Regression",
        "mse"        : round(mse, 4),
        "rmse"       : round(rmse, 4),
        "r2"         : round(r2, 4),
        "scores"     : scores
    }


def get_coefficients(model: LinearRegression, feature_names: list) -> dict:
    """
    Returns the regression coefficients for each feature.
    Larger absolute coefficient = stronger influence on risk score.
    Useful for explaining which features drive the prediction.
    """
    return dict(zip(feature_names, model.coef_))


def save_model(model: LinearRegression, path: str = "models/stage1_regression.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Stage 1] Model saved to {path}")


def load_model(path: str = "models/stage1_regression.pkl") -> LinearRegression:
    return joblib.load(path)
