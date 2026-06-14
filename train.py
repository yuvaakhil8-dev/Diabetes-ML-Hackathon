# train.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Master Training Script
#
# Run this ONCE before launching the dashboard:
#   python train.py
#
# What it does:
#   1. Loads and preprocesses the dataset
#   2. Trains all three pipeline stages
#   3. Evaluates every model on the test set
#   4. Saves all trained models to models/ folder
#   5. Prints a final metrics comparison table
#
# After this runs successfully, launch the UI with:
#   streamlit run app/dashboard.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import os
import sys
import joblib
import numpy as np

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess      import load_and_clean, split_and_scale, save_scaler
from src.stage1_regression import train as train_reg, evaluate as eval_reg, save_model as save_reg
from src.stage2_tree       import (train as train_tree, compare_criteria,
                                   compute_information_gain, get_feature_importances,
                                   save_model as save_tree)
from src.stage3_knn        import find_best_k, save_models as save_knn
from src.evaluate          import print_metrics_table

DATA_PATH = "data/diabetes.csv"


def main():
    print("\n" + "â•"*60)
    print("  DiabetaSense â€” ML Training Pipeline")
    print("  Progressive Patient Risk Assessment")
    print("â•"*60 + "\n")

    # â”€â”€ Check dataset exists â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at '{DATA_PATH}'")
        print("  Please download 'diabetes.csv' from Kaggle and place it in the data/ folder.")
        print("  Link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        sys.exit(1)

    # â”€â”€ Stage 0: Preprocessing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("â”€â”€ Stage 0: Preprocessing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    df = load_and_clean(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)
    save_scaler(scaler)

    # Save training data for KNN neighbor lookup in the UI
    os.makedirs("models", exist_ok=True)
    joblib.dump((X_train, y_train), "models/train_data.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    print()

    # â”€â”€ Stage 1: Linear Regression â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("â”€â”€ Stage 1: Linear Regression â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    reg_model     = train_reg(X_train, y_train)
    reg_results   = eval_reg(reg_model, X_test, y_test)
    save_reg(reg_model)
    print()

    # â”€â”€ Stage 2: Decision Tree â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("â”€â”€ Stage 2: Decision Tree â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    best_tree, best_tree_results, gini_results, entropy_results = compare_criteria(
        X_train, y_train, X_test, y_test
    )

    # Information Gain (our manual computation)
    ig_scores = compute_information_gain(X_train, y_train, feature_names)
    print("\n  Information Gain per feature:")
    for feat, ig in ig_scores.items():
        print(f"    {feat:<30} {ig:.4f}")

    # Feature importances from the trained tree
    fi_scores = get_feature_importances(best_tree, feature_names)
    print("\n  Tree Feature Importances (sklearn):")
    for feat, imp in fi_scores.items():
        print(f"    {feat:<30} {imp:.4f}")

    # Save all tree-related data
    save_tree(best_tree)
    joblib.dump(ig_scores,      "models/ig_scores.pkl")
    joblib.dump(fi_scores,      "models/fi_scores.pkl")
    joblib.dump(gini_results,   "models/gini_results.pkl")
    joblib.dump(entropy_results,"models/entropy_results.pkl")
    print()

    # â”€â”€ Stage 3: KNN + Weighted KNN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("â”€â”€ Stage 3: KNN + Weighted KNN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    knn_output = find_best_k(X_train, y_train, X_test, y_test)
    save_knn(knn_output["best_knn_model"], knn_output["best_wknn_model"])
    joblib.dump(knn_output["k_results"], "models/k_results.pkl")
    print()

    # â”€â”€ Final Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("â”€â”€ Final Metrics Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    all_results = [
        best_tree_results,
        knn_output["best_knn_results"],
        knn_output["best_wknn_results"]
    ]
    print_metrics_table(all_results)

    print("[train.py] All models saved successfully.")
    print("[train.py] Now run:  streamlit run app/dashboard.py\n")


if __name__ == "__main__":
    main()
