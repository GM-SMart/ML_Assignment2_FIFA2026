"""
app.py - Streamlit app for ML Assignment 2
FIFA World Cup 2026: High-Performer Classification (5 ML models)

Features:
  a. CSV upload of test data
  b. Model selection dropdown
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report
"""
import json
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

st.set_page_config(page_title="FIFA WC 2026 – ML Classifier", page_icon="⚽", layout="wide")


# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    models = {name: joblib.load(os.path.join(MODEL_DIR, fn))
              for name, fn in meta["model_files"].items()}
    return meta, models


def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", c.strip().lower())).strip("_")
                  for c in df.columns]
    return df


def scores(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def compute_metrics(y, y_pred, y_score):
    auc = roc_auc_score(y, y_score) if len(np.unique(y)) == 2 else float("nan")
    return {"Accuracy": accuracy_score(y, y_pred), "AUC": auc,
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred)}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
try:
    meta, models = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python model/train_models.py` first.")
    st.stop()

features = meta["numeric_features"] + meta["categorical_features"]
target, rating_col, thr = meta["target"], meta["rating_column"], meta["threshold"]

st.title("⚽ FIFA World Cup 2026 – High-Performer Classification")
st.caption(f"Predicts whether a player's match `{rating_col}` ≥ {thr:.2f} (dataset median) "
           "using 5 ML classifiers trained on the same data.")

with st.sidebar:
    st.header("1️⃣ Upload test data")
    uploaded = st.file_uploader("Upload CSV (test data only)", type=["csv"])
    sample = os.path.join(BASE, "test_data.csv")
    if os.path.exists(sample):
        with open(sample, "rb") as f:
            st.download_button("⬇️ Download sample test_data.csv", f, "test_data.csv", "text/csv")
    st.header("2️⃣ Select model")
    model_name = st.selectbox("Classification model", list(models.keys()))
    st.markdown("---")
    st.caption(f"Training rows: {meta['train_rows']:,} | Test rows: {meta['test_rows']:,} | "
               f"Features: {len(features)}")

tab_eval, tab_cmp, tab_info = st.tabs(["📊 Evaluate on uploaded data",
                                       "🏆 Model comparison (hold-out test set)",
                                       "ℹ️ Dataset & features"])

with tab_eval:
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar (for example, the provided `test_data.csv`) to see "
                "predictions, evaluation metrics, a confusion matrix and a classification report.")
    else:
        df = clean_columns(pd.read_csv(uploaded))
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"The uploaded file is missing {len(missing)} required feature column(s): "
                     f"{missing}")
            st.stop()

        # ground truth: label column, or derive it from the rating column
        y = None
        if target in df.columns:
            y = df[target].astype(int)
        elif rating_col in df.columns:
            y = (df[rating_col] >= thr).astype(int)

        X = df[features]
        model = models[model_name]
        y_pred = model.predict(X)
        y_score = scores(model, X)
        st.subheader(f"Model: {model_name}  —  {len(df):,} rows")

        if y is None:
            st.warning(f"No `{target}` or `{rating_col}` column found, so only predictions are shown "
                       "(metrics need true labels).")
        else:
            m = compute_metrics(y, y_pred, y_score)
            cols = st.columns(6)
            for c, (k, v) in zip(cols, m.items()):
                c.metric(k, f"{v:.4f}")

            left, right = st.columns(2)
            with left:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y, y_pred, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                            xticklabels=["Pred 0 (Low)", "Pred 1 (High)"],
                            yticklabels=["True 0 (Low)", "True 1 (High)"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig)
            with right:
                st.markdown("**Classification Report**")
                rep = classification_report(y, y_pred, labels=[0, 1],
                                            target_names=["Low (0)", "High (1)"],
                                            output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(rep).T.round(4))

            if st.checkbox("Compare all 5 models on the uploaded data"):
                allm = pd.DataFrame({n: compute_metrics(y, mdl.predict(X), scores(mdl, X))
                                     for n, mdl in models.items()}).T
                st.dataframe(allm.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"))

        out = df.copy()
        out["predicted_" + target] = y_pred
        st.markdown("**Predictions (first 100 rows)**")
        st.dataframe(out.head(100))
        st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")

with tab_cmp:
    mpath = os.path.join(MODEL_DIR, "metrics.csv")
    if os.path.exists(mpath):
        mt = pd.read_csv(mpath, index_col=0)
        st.dataframe(mt.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"))
        fig, ax = plt.subplots(figsize=(10, 4))
        mt.plot(kind="bar", ax=ax, rot=0)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend(ncol=6, loc="lower center")
        st.pyplot(fig)
        best = mt.sort_values(["MCC", "F1", "AUC"], ascending=False).index[0]
        st.success(f"🏆 Best model by MCC: **{best}** (MCC = {mt.loc[best, 'MCC']:.4f})")

with tab_info:
    st.markdown(f"**Dataset:** `{meta['dataset']}` — {meta['n_rows']:,} rows")
    st.markdown(f"**Target:** `{target}` = 1 if `{rating_col}` ≥ {thr:.2f}; class balance "
                f"{meta['class_balance']}")
    st.markdown(f"**Numeric features ({len(meta['numeric_features'])}):** "
                + ", ".join(meta["numeric_features"]))
    st.markdown(f"**Categorical features ({len(meta['categorical_features'])}):** "
                + ", ".join(meta["categorical_features"]))
    st.markdown("**Dropped columns:**")
    st.json(meta["dropped_columns"])
    st.markdown("**Tuned hyper-parameters:**")
    st.json(meta["best_params"])
