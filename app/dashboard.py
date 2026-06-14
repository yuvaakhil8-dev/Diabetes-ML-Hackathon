# app/dashboard.py
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DiabetaSense â€” Streamlit Dashboard
#
# Run with:
#   streamlit run app/dashboard.py
#
# Prerequisites:
#   Run train.py first so all model .pkl files exist in models/
#
# Pages:
#   1. Patient Assessment  â€” enter patient data, run pipeline, get result
#   2. Model Metrics       â€” comparison of all models on test set
#   3. Stage Analysis      â€” deep dive into each stage's charts
#   4. About               â€” pipeline explanation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import os
import sys
import numpy as np
import pandas as pd
import joblib
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stage1_regression import predict_risk_scores, apply_threshold, THRESHOLD
from src.stage3_knn        import get_neighbors
from src.evaluate          import (
    plot_confusion_matrix, plot_information_gain,
    plot_k_vs_accuracy, plot_model_comparison,
    plot_risk_score_histogram
)

# â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="DiabetaSense",
    page_icon="ðŸ©º",
    layout="wide",
    initial_sidebar_state="expanded"
)

# â”€â”€ Custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stButton>button {
        background-color: #1D9E75; color: white;
        border-radius: 8px; border: none;
        padding: 0.5rem 2rem; font-size: 1rem; font-weight: 600;
    }
    .stButton>button:hover { background-color: #166b50; }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem; margin: 0.3rem 0;
        border-left: 4px solid #1D9E75;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .risk-high {
        background: #FAECE7; border-left: 4px solid #D85A30;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 1.1rem; font-weight: 600; color: #4A1B0C;
    }
    .risk-low {
        background: #E1F5EE; border-left: 4px solid #1D9E75;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 1.1rem; font-weight: 600; color: #085041;
    }
    .stage-label {
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
        color: #7F77DD; text-transform: uppercase; margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# â”€â”€ Load all models and cached data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_resource
def load_all():
    """Load all saved models and supporting data once."""
    base = os.path.join(os.path.dirname(__file__), "..", "models")

    def p(name): return os.path.join(base, name)

    missing = [f for f in [
        "scaler.pkl", "stage1_regression.pkl", "stage2_tree.pkl",
        "stage3_knn.pkl", "stage3_wknn.pkl", "train_data.pkl",
        "feature_names.pkl", "ig_scores.pkl", "fi_scores.pkl",
        "gini_results.pkl", "entropy_results.pkl", "k_results.pkl"
    ] if not os.path.exists(p(f))]

    if missing:
        return None, missing

    return {
        "scaler"          : joblib.load(p("scaler.pkl")),
        "reg_model"       : joblib.load(p("stage1_regression.pkl")),
        "tree_model"      : joblib.load(p("stage2_tree.pkl")),
        "knn_model"       : joblib.load(p("stage3_knn.pkl")),
        "wknn_model"      : joblib.load(p("stage3_wknn.pkl")),
        "X_train"         : joblib.load(p("train_data.pkl"))[0],
        "y_train"         : joblib.load(p("train_data.pkl"))[1],
        "feature_names"   : joblib.load(p("feature_names.pkl")),
        "ig_scores"       : joblib.load(p("ig_scores.pkl")),
        "fi_scores"       : joblib.load(p("fi_scores.pkl")),
        "gini_results"    : joblib.load(p("gini_results.pkl")),
        "entropy_results" : joblib.load(p("entropy_results.pkl")),
        "k_results"       : joblib.load(p("k_results.pkl")),
    }, []


data, missing_files = load_all()

# â”€â”€ Guard: models not trained yet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if data is None:
    st.error("âš ï¸ Models not found. Please run `python train.py` first.")
    st.code("python train.py", language="bash")
    st.stop()


# â”€â”€ Sidebar navigation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.sidebar.image("https://img.icons8.com/color/96/stethoscope.png", width=64)
st.sidebar.title("DiabetaSense")
st.sidebar.caption("Progressive Patient Risk Assessment")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["ðŸ©º Patient Assessment", "ðŸ“Š Model Metrics", "ðŸ”¬ Stage Analysis", "â„¹ï¸ About"]
)

st.sidebar.markdown("---")
openrouter_key = st.sidebar.text_input(
    "OpenRouter API Key (optional)",
    type="password",
    help="Paste your free OpenRouter API key to get AI-generated plain-English explanations."
)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 1 â€” Patient Assessment
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if page == "ðŸ©º Patient Assessment":
    st.title("ðŸ©º Patient Risk Assessment")
    st.markdown("Enter the patient's health data below. The pipeline runs all three stages and gives a final result.")
    st.markdown("---")

    # â”€â”€ Input form â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Pregnancies",           0, 17, 3)
        glucose     = st.slider("Glucose (mg/dL)",       44, 199, 120)
        bp          = st.slider("Blood Pressure (mm Hg)", 24, 122, 72)
        skin        = st.slider("Skin Thickness (mm)",    7, 99, 23)
    with col2:
        insulin     = st.slider("Insulin (mu U/ml)",     14, 846, 80)
        bmi         = st.slider("BMI",                   18.0, 67.0, 32.0, step=0.1)
        dpf         = st.slider("Diabetes Pedigree",     0.08, 2.42, 0.47, step=0.01)
        age         = st.slider("Age",                   21, 81, 33)

    st.markdown("---")
    run_btn = st.button("Run Pipeline â†’")

    if run_btn:
        # â”€â”€ Assemble and scale input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        raw_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled_input = data["scaler"].transform(raw_input)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # STAGE 1 â€” Screening
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        st.markdown("### Pipeline Results")
        s1, s2, s3 = st.columns(3)

        risk_score   = predict_risk_scores(data["reg_model"], scaled_input)[0]
        risk_label   = apply_threshold(np.array([risk_score]))[0]

        with s1:
            st.markdown('<div class="stage-label">Stage 1 â€” Screening</div>', unsafe_allow_html=True)
            st.metric("Risk Score", f"{risk_score:.3f}", help="Linear Regression output (0â€“1)")
            if risk_label == 1:
                st.markdown('<div class="risk-high">âš ï¸ High Risk â€” advancing to diagnosis</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">âœ… Low Risk â€” no further escalation</div>',
                            unsafe_allow_html=True)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # STAGE 2 â€” Decision Tree diagnosis (all patients shown)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        tree_pred  = data["tree_model"].predict(scaled_input)[0]
        tree_proba = data["tree_model"].predict_proba(scaled_input)[0][1]

        with s2:
            st.markdown('<div class="stage-label">Stage 2 â€” Diagnosis</div>', unsafe_allow_html=True)
            st.metric("Decision Tree", "Diabetic" if tree_pred == 1 else "Non-Diabetic",
                      delta=f"Confidence: {tree_proba:.2%}")

            # Show which feature matters most for this patient
            top_feature = list(data["fi_scores"].keys())[0]
            st.caption(f"Most influential feature: **{top_feature}**")

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # STAGE 3 â€” KNN + Weighted KNN validation
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        knn_pred   = data["knn_model"].predict(scaled_input)[0]
        wknn_pred  = data["wknn_model"].predict(scaled_input)[0]
        wknn_proba = data["wknn_model"].predict_proba(scaled_input)[0][1]

        with s3:
            st.markdown('<div class="stage-label">Stage 3 â€” Validation</div>', unsafe_allow_html=True)
            st.metric("KNN",          "Diabetic" if knn_pred  == 1 else "Non-Diabetic")
            st.metric("Weighted KNN", "Diabetic" if wknn_pred == 1 else "Non-Diabetic",
                      delta=f"Confidence: {wknn_proba:.2%}")

        # â”€â”€ Consensus â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        votes     = tree_pred + knn_pred + wknn_pred
        consensus = "Diabetic" if votes >= 2 else "Non-Diabetic"
        consensus_color = "#FAECE7" if consensus == "Diabetic" else "#E1F5EE"
        consensus_icon  = "ðŸ”´" if consensus == "Diabetic" else "ðŸŸ¢"

        st.markdown("---")
        st.markdown(
            f'<div style="background:{consensus_color};border-radius:10px;padding:1rem 1.5rem;">'
            f'<span style="font-size:1.3rem;font-weight:700;">{consensus_icon} Final Consensus: {consensus}</span>'
            f'<br><span style="font-size:0.9rem;color:#555;">'
            f'Stage 2 (DT): {"Diabetic" if tree_pred==1 else "Non-Diabetic"} | '
            f'Stage 3 (KNN): {"Diabetic" if knn_pred==1 else "Non-Diabetic"} | '
            f'Stage 3 (WKNN): {"Diabetic" if wknn_pred==1 else "Non-Diabetic"}'
            f'</span></div>',
            unsafe_allow_html=True
        )

        # â”€â”€ Similar past patients (KNN neighbors) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown("---")
        st.markdown("#### Similar Historical Patients (from training set)")
        neighbors = get_neighbors(
            data["wknn_model"], data["X_train"], data["y_train"],
            scaled_input[0], data["feature_names"]
        )
        neighbor_df = pd.DataFrame([{
            "Rank": n["rank"],
            "Distance": n["distance"],
            "Outcome": n["outcome"]
        } for n in neighbors])
        st.dataframe(neighbor_df, use_container_width=True, hide_index=True)

        # â”€â”€ OpenRouter AI Explanation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown("---")
        st.markdown("#### text report")

        if openrouter_key:
            with st.spinner("Generating explanation..."):
                prompt = (
                    f"A patient has the following health data: "
                    f"Pregnancies={pregnancies}, Glucose={glucose} mg/dL, "
                    f"Blood Pressure={bp} mm Hg, Skin Thickness={skin} mm, "
                    f"Insulin={insulin} mu U/ml, BMI={bmi}, "
                    f"Diabetes Pedigree Function={dpf}, Age={age} years. "
                    f"A machine learning pipeline assessed them as follows: "
                    f"Risk Score={risk_score:.3f} (threshold=0.5), "
                    f"Decision Tree says '{consensus}', "
                    f"KNN and Weighted KNN both agree. "
                    f"The most important feature for this prediction was {top_feature}. "
                    f"Please explain this result in simple, clear language that a patient "
                    f"(not a doctor) can understand. Keep it under 120 words. "
                    f"Do not give medical advice. End with one practical lifestyle tip."
                )
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "mistralai/mistral-7b-instruct:free",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 200
                        },
                        timeout=15
                    )
                    explanation = response.json()["choices"][0]["message"]["content"]
                    st.info(explanation)
                except Exception as e:
                    st.warning(f"Could not fetch explanation: {e}")
        else:
            st.caption("Add your OpenRouter API key in the sidebar to get a plain-English explanation of this result.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 2 â€” Model Metrics
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ“Š Model Metrics":
    st.title("ðŸ“Š Model Metrics â€” Test Set Evaluation")
    st.markdown("All metrics computed on the held-out 20% test set.")
    st.markdown("---")

    gini    = data["gini_results"]
    entropy = data["entropy_results"]
    k_res   = data["k_results"]

    # Build comparison table
    rows = []
    for r in [gini, entropy]:
        rows.append({
            "Model"    : r["model_name"],
            "Accuracy" : r["accuracy"],
            "Precision": r["precision"],
            "Recall"   : r["recall"],
            "F1 Score" : r["f1"],
            "ROC-AUC"  : r["roc_auc"]
        })

    # Best KNN and WKNN
    best_k_knn  = k_res["k_values"][int(np.argmax(k_res["knn_f1"]))]
    best_k_wknn = k_res["k_values"][int(np.argmax(k_res["wknn_f1"]))]

    rows.append({
        "Model"    : f"KNN (k={best_k_knn})",
        "Accuracy" : k_res["knn_accuracy"][int(np.argmax(k_res["knn_f1"]))],
        "Precision": "â€”", "Recall": "â€”",
        "F1 Score" : max(k_res["knn_f1"]),
        "ROC-AUC"  : "â€”"
    })
    rows.append({
        "Model"    : f"Weighted KNN (k={best_k_wknn})",
        "Accuracy" : k_res["wknn_accuracy"][int(np.argmax(k_res["wknn_f1"]))],
        "Precision": "â€”", "Recall": "â€”",
        "F1 Score" : max(k_res["wknn_f1"]),
        "ROC-AUC"  : "â€”"
    })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Model Comparison Chart")
    metric_choice = st.selectbox("Compare by:", ["f1", "accuracy", "precision", "recall", "roc_auc"])

    all_results = [gini, entropy]
    img = plot_model_comparison(all_results, metric=metric_choice)
    st.image(img, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Confusion Matrices")
    c1, c2 = st.columns(2)
    with c1:
        st.image(plot_confusion_matrix(gini["confusion_matrix"],    "Decision Tree â€” Gini"),    use_container_width=True)
    with c2:
        st.image(plot_confusion_matrix(entropy["confusion_matrix"], "Decision Tree â€” Entropy"), use_container_width=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 3 â€” Stage Analysis
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ”¬ Stage Analysis":
    st.title("ðŸ”¬ Stage-by-Stage Analysis")

    tab1, tab2, tab3 = st.tabs(["Stage 1 â€” Regression", "Stage 2 â€” Decision Tree", "Stage 3 â€” KNN"])

    with tab1:
        st.markdown("#### Risk Score Distribution")
        st.markdown(
            "This chart shows how the linear regression risk scores are distributed "
            "across actual diabetic and non-diabetic patients in the test set. "
            "A good model separates the two distributions clearly around the threshold."
        )
        # We don't store test scores separately â€” recompute from gini predictions
        # Use gini y_pred as a proxy for the flag (approximation for display)
        gini = data["gini_results"]
        st.caption("Note: Risk scores are continuous outputs from Linear Regression before thresholding.")

        st.markdown("#### Linear Regression Coefficients")
        st.markdown(
            "Each coefficient shows how much the risk score increases per unit increase "
            "in that feature (after normalization). Larger positive values = stronger positive influence on risk."
        )
        reg_model = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "stage1_regression.pkl"))
        coef_df = pd.DataFrame({
            "Feature"    : data["feature_names"],
            "Coefficient": reg_model.coef_
        }).sort_values("Coefficient", ascending=False)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### Information Gain per Feature")
        st.markdown(
            "Information Gain measures how much each feature reduces uncertainty (entropy) "
            "when used as a split point. Higher = more useful for diagnosis."
        )
        st.image(plot_information_gain(data["ig_scores"]), use_container_width=True)

        st.markdown("#### Feature Importance (sklearn â€” from trained tree)")
        fi_df = pd.DataFrame({
            "Feature"   : list(data["fi_scores"].keys()),
            "Importance": list(data["fi_scores"].values())
        })
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

        st.markdown("#### Gini vs Entropy â€” Head-to-head")
        g = data["gini_results"]
        e = data["entropy_results"]
        compare_df = pd.DataFrame({
            "Metric"   : ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "Gini"     : [g["accuracy"], g["precision"], g["recall"], g["f1"], g["roc_auc"]],
            "Entropy"  : [e["accuracy"], e["precision"], e["recall"], e["f1"], e["roc_auc"]],
        })
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Accuracy across K values â€” KNN vs Weighted KNN")
        st.markdown(
            "This chart shows how accuracy changes as K increases for both variants. "
            "The best K is chosen based on the highest F1 score, not just accuracy."
        )
        st.image(plot_k_vs_accuracy(data["k_results"]), use_container_width=True)

        st.markdown("#### Why Weighted KNN?")
        st.markdown(
            "In standard KNN, all K neighbors vote equally. In Weighted KNN, "
            "a neighbor at distance 0.1 has 10x more influence than one at distance 1.0. "
            "This is more realistic â€” a very similar patient should count more than a barely-similar one."
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 4 â€” About
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "â„¹ï¸ About":
    st.title("â„¹ï¸ About DiabetaSense")
    st.markdown("""
**DiabetaSense** is a three-stage Progressive Patient Risk Assessment Pipeline built for the ML Hackathon â€” 2nd Lab Evaluation, April 4, 2026.

---

### Pipeline Overview

| Stage | Model | Role |
|---|---|---|
| Stage 1 | Linear Regression | Screens every patient, outputs a risk score |
| Stage 2 | Decision Tree | Diagnoses high-risk patients, explains which features drove the result |
| Stage 3 | KNN + Weighted KNN | Validates the diagnosis by finding similar historical patients |

---

### Dataset
Pima Indians Diabetes Dataset â€” 768 patients, 8 features, binary outcome.
Source: UCI Machine Learning Repository / Kaggle.

---

### Concepts Covered
- Linear Regression, MSE, RMSE, RÂ²
- Decision Tree, Entropy, Gini Impurity, Information Gain
- KNN, Weighted KNN, Euclidean Distance
- Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

---

### Team
2nd Year B.Tech AIE â€” ML Hackathon, April 2026.
    """)
