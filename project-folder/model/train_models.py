"""
train_models.py
===============
ML Assignment 2 - FIFA World Cup 2026 Player Performance (binary classification)

What this script does (end-to-end, reproducible):
  1. Downloads the dataset with kagglehub (or uses --data <csv path>).
  2. Cleans column names and defines the binary target:
         high_performer = 1  if player_rating >= median(player_rating)  else 0
  3. Removes identifier columns, high-cardinality text columns and every column
     that would leak the target (rating-derived / post-match composite scores,
     tournament-total aggregates, Player-of-the-Match flag).
  4. Splits the data 80/20 (stratified, random_state=42) and saves the 20% hold-out
     split as test_data.csv (used by the Streamlit app).
  5. Trains the 5 required classifiers inside identical preprocessing pipelines:
         Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM
  6. Computes Accuracy, AUC, Precision, Recall, F1 and MCC on the hold-out set.
  7. Saves every trained pipeline (.pkl), the preprocessing artifacts, the metrics
     table, and re-writes README.md with the real comparison table + observations.

Run from the project root:
    python model/train_models.py
    python model/train_models.py --data path/to/fifa_world_cup_2026_player_performance.csv
"""

import argparse
import warnings
import glob
import json
import os
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)   # harmless numpy cast warning in GridSearchCV

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
KAGGLE_DATASET = "rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_NAME = "high_performer"          # name of the derived binary label
MAX_CATEGORIES = 15                     # text columns with more levels are dropped
LEAKAGE_CORR_LIMIT = 0.90               # safety net: |corr| with rating above this -> dropped
SVM_MAX_TRAIN_ROWS = 20000              # RBF-SVM is O(n^2); a stratified subsample keeps it fast

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODEL_DIR)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}

# Column-name patterns that leak the target (derived from / computed after the rating)
LEAKAGE_PATTERNS = [
    r"rating",                     # player_rating (source of target), tournament_rating
    r"performance_score",
    r"contribution",               # offensive_/defensive_contribution (composite scores)
    r"possession_impact", r"pressure_resistance", r"creativity_score",
    r"consistency_score", r"clutch",
    r"player_of_the_match", r"potm", r"motm", r"man_of_the_match",
    r"^total_.*_tournament$", r"_tournament$",   # whole-tournament aggregates (future info)
]
# Identifier / free-text columns that carry no generalisable signal
ID_PATTERNS = [r"(^|_)id$", r"(^|_)name$", r"date", r"time$"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """lower-case snake_case column names, e.g. 'Pass Accuracy (%)' -> 'pass_accuracy'."""
    df = df.copy()
    df.columns = [re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", c.strip().lower())).strip("_")
                  for c in df.columns]
    return df


def locate_csv(data_arg):
    """Return the path of the dataset CSV (downloaded with kagglehub if needed)."""
    if data_arg:
        return data_arg
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Path to dataset files:", path)
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        sys.exit("No CSV file found in the downloaded dataset folder.")
    return max(csvs, key=os.path.getsize)            # main data file = largest CSV


def find_rating_column(cols, override=None):
    if override:
        return override
    if "player_rating" in cols:
        return "player_rating"
    cands = [c for c in cols if "rating" in c and "tournament" not in c]
    if not cands:
        sys.exit(f"Could not find a rating column. Columns are:\n{list(cols)}\n"
                 "Re-run with --rating-col <column>.")
    return cands[0]


def matches_any(col, patterns):
    return any(re.search(p, col) for p in patterns)


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def positive_scores(model, X):
    """Probability of class 1 (or SVM decision value) - used for AUC."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to the dataset CSV (skip kagglehub download)")
    ap.add_argument("--rating-col", help="Override the rating column name")
    args = ap.parse_args()

    # 1. Load ---------------------------------------------------------------
    csv_path = locate_csv(args.data)
    print("Using file:", csv_path)
    raw = clean_columns(pd.read_csv(csv_path))
    print(f"Raw shape: {raw.shape}")

    # 2. Target -------------------------------------------------------------
    rating_col = find_rating_column(raw.columns, args.rating_col)
    raw = raw.dropna(subset=[rating_col]).drop_duplicates().reset_index(drop=True)
    threshold = float(raw[rating_col].median())
    raw[TARGET_NAME] = (raw[rating_col] >= threshold).astype(int)
    print(f"Target: {TARGET_NAME} = 1 if {rating_col} >= {threshold:.4f} (median)")
    print("Class balance:", raw[TARGET_NAME].value_counts(normalize=True).round(4).to_dict())

    # 3. Feature selection --------------------------------------------------
    dropped = {"target_source / leakage": [], "identifier": [], "high_cardinality_text": [],
               "constant": [], "high_corr_with_rating": []}
    features = []
    for c in raw.columns:
        if c == TARGET_NAME:
            continue
        if c == rating_col or matches_any(c, LEAKAGE_PATTERNS):
            dropped["target_source / leakage"].append(c)
        elif matches_any(c, ID_PATTERNS):
            dropped["identifier"].append(c)
        elif raw[c].nunique(dropna=True) <= 1:
            dropped["constant"].append(c)
        elif (raw[c].dtype == object or str(raw[c].dtype) == "category") and \
                raw[c].nunique() > MAX_CATEGORIES:
            dropped["high_cardinality_text"].append(c)
        else:
            features.append(c)

    X_all = raw[features].copy()
    for c in X_all.columns:                       # booleans -> 0/1
        if X_all[c].dtype == bool:
            X_all[c] = X_all[c].astype(int)
    num_cols = X_all.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X_all.columns if c not in num_cols]

    # safety net against unknown leakage columns
    corr = X_all[num_cols].corrwith(raw[rating_col]).abs()
    leak = corr[corr >= LEAKAGE_CORR_LIMIT].index.tolist()
    if leak:
        dropped["high_corr_with_rating"] = leak
        num_cols = [c for c in num_cols if c not in leak]
    features = num_cols + cat_cols
    X_all = X_all[features]
    y_all = raw[TARGET_NAME]

    print(f"\nFeatures used ({len(features)}): {len(num_cols)} numeric, {len(cat_cols)} categorical")
    for k, v in dropped.items():
        if v:
            print(f"  dropped [{k}]: {v}")

    # 4. Split --------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE)
    print(f"\nTrain: {X_train.shape}   Test: {X_test.shape}")

    # test_data.csv keeps the raw rating + label so the app can compute metrics
    test_export = raw.loc[X_test.index, features + [rating_col, TARGET_NAME]]
    test_export.to_csv(os.path.join(ROOT_DIR, "test_data.csv"), index=False)

    # 5. Preprocessing (fit only on training data, inside each pipeline) -----
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ], sparse_threshold=0)   # dense output (required by GaussianNB)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    specs = {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"clf__max_depth": [4, 6, 8, 10, 12],
                           "clf__min_samples_leaf": [1, 20, 50]}),
        "KNN": (KNeighborsClassifier(n_jobs=-1),
                {"clf__n_neighbors": [11, 21, 35, 51]}),
        "Naive Bayes": (GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]}),
        "Support Vector Machine": (SVC(kernel="rbf", C=1.0, gamma="scale",
                                       random_state=RANDOM_STATE), None),
    }

    results, train_acc, best_params = {}, {}, {}
    for name, (clf, grid) in specs.items():
        t0 = time.time()
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        Xf, yf = X_train, y_train
        if name == "Support Vector Machine" and len(X_train) > SVM_MAX_TRAIN_ROWS:
            Xf, _, yf, _ = train_test_split(X_train, y_train, train_size=SVM_MAX_TRAIN_ROWS,
                                            stratify=y_train, random_state=RANDOM_STATE)
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
            gs.fit(Xf, yf)
            model, best_params[name] = gs.best_estimator_, gs.best_params_
        else:
            model = pipe.fit(Xf, yf)
            best_params[name] = {"C": 1.0, "kernel": "rbf", "gamma": "scale",
                                 "train_rows": int(len(Xf))}
        y_pred = model.predict(X_test)
        results[name] = evaluate(y_test, y_pred, positive_scores(model, X_test))
        train_acc[name] = accuracy_score(y_train, model.predict(X_train))
        joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILES[name]), compress=3)
        print(f"{name:<24} done in {time.time() - t0:6.1f}s  "
              f"test acc={results[name]['Accuracy']:.4f}  params={best_params[name]}")

    # 6. Save metrics + artifacts ------------------------------------------
    metrics = pd.DataFrame(results).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    metrics.index.name = "ML Model Name"
    metrics.round(4).to_csv(os.path.join(MODEL_DIR, "metrics.csv"))
    print("\n", metrics.round(4).to_string())

    joblib.dump(preprocessor.fit(X_train), os.path.join(MODEL_DIR, "preprocessor.pkl"), compress=3)
    meta = {
        "dataset": KAGGLE_DATASET, "source_file": os.path.basename(csv_path),
        "n_rows": int(len(raw)), "n_raw_columns": int(raw.shape[1] - 1),
        "rating_column": rating_col, "target": TARGET_NAME, "threshold": threshold,
        "class_balance": {str(k): float(v) for k, v in y_all.value_counts(normalize=True).items()},
        "numeric_features": num_cols, "categorical_features": cat_cols,
        "dropped_columns": dropped, "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "best_params": {k: {kk: (vv if isinstance(vv, (int, float, str)) else str(vv))
                            for kk, vv in v.items()} for k, v in best_params.items()},
        "train_accuracy": train_acc, "model_files": MODEL_FILES,
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "joblib": joblib.__version__},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    write_requirements(meta["versions"])
    write_readme(metrics, meta)
    print("\nSaved models, metrics.csv, metadata.json, test_data.csv, requirements.txt, README.md")


# ----------------------------------------------------------------------------
# requirements.txt (pins scikit-learn so pickled models load on Streamlit Cloud)
# ----------------------------------------------------------------------------
def write_requirements(v):
    np_major = int(v["numpy"].split(".")[0])
    lines = ["streamlit", f"scikit-learn=={v['scikit-learn']}",
             "numpy>=2.0" if np_major >= 2 else "numpy",   # numpy 2 can load numpy-1 pickles
             "pandas", "matplotlib", "seaborn", "joblib", "kagglehub"]
    with open(os.path.join(ROOT_DIR, "requirements.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# README.md generation (real numbers + data-driven observations)
# ----------------------------------------------------------------------------
def observations(m: pd.DataFrame, meta: dict) -> dict:
    tr = meta["train_accuracy"]
    rank = m["MCC"].rank(ascending=False, method="min").astype(int)
    n = len(m)
    best = m.sort_values(["MCC", "F1", "AUC"], ascending=False).index[0]
    lr, dt, knn, nb, svm = (m.loc[k] for k in ["Logistic Regression", "Decision Tree", "KNN",
                                                 "Naive Bayes", "Support Vector Machine"])
    nfeat = len(meta["numeric_features"]) + len(meta["categorical_features"])

    def head(k):
        return (f"Rank {rank[k]}/{n} by MCC (train acc {tr[k]:.4f} vs test acc "
                f"{m.loc[k, 'Accuracy']:.4f}, gap {tr[k] - m.loc[k, 'Accuracy']:+.4f}). ")

    def params(k):
        return ", ".join(f"{kk.replace('clf__', '')}={vv}" for kk, vv in meta["best_params"][k].items()
                         if kk != "train_rows")

    def gap_txt(k):
        g = tr[k] - m.loc[k, "Accuracy"]
        if g > 0.05:
            return "The large train-test gap shows overfitting. "
        if g > 0.02:
            return "A small train-test gap shows mild overfitting. "
        return "Train and test scores are close, so it generalises well (no overfitting). "

    o = {}
    o["Logistic Regression"] = head("Logistic Regression") + gap_txt("Logistic Regression") + (
        "This linear model is as good as or better than the non-linear models, so the rating is driven "
        "mostly by additive effects of goals, assists, chance creation and passing quality. "
        if lr.MCC >= max(dt.MCC, knn.MCC, svm.MCC) - 0.01 else
        "It is beaten by the non-linear models, which means interactions between the statistics "
        "(for example, position combined with defensive actions) matter. A single linear boundary misses these. ") + (
        f"Precision ({lr.Precision:.3f}) and recall ({lr.Recall:.3f}) are balanced."
        if abs(lr.Precision - lr.Recall) < 0.05 else
        f"It favours {'precision' if lr.Precision > lr.Recall else 'recall'} "
        f"(P={lr.Precision:.3f}, R={lr.Recall:.3f}).")
    o["Decision Tree"] = head("Decision Tree") + gap_txt("Decision Tree") + (
        f"The tuned tree ({params('Decision Tree')}) is pruned to limit variance. "
        "Axis-aligned threshold splits capture non-linear rules (for example, goals ≥ 1 together with a high pass accuracy), "
        + ("so it beats the linear baseline."
           if dt.MCC > lr.MCC else
           "but its step-wise boundaries are coarser than a smooth linear score, so it does worse than Logistic Regression."))
    o["KNN"] = head("KNN") + gap_txt("KNN") + (
        f"KNN ({params('KNN')}) relies on distances in a {nfeat}-feature scaled space. "
        + ("It is competitive here, so performance profiles form clear local neighbourhoods. "
           if knn.MCC >= lr.MCC - 0.02 else
           "Many weak or irrelevant dimensions dilute the distance (the curse of dimensionality), "
           "so it does worse than the model-based classifiers. ")
        + "It is also the slowest model at prediction time because it stores the entire training set.")
    o["Naive Bayes"] = head("Naive Bayes") + gap_txt("Naive Bayes") + (
        "Gaussian NB assumes the features are independent and normally distributed. Football statistics break both assumptions: "
        "shots, shots on target and xG are strongly correlated, and counts such as goals are skewed and non-Gaussian. "
        + ("This makes it the weakest model, " if rank["Naive Bayes"] == n else "Even so, it stays reasonably robust, ")
        + (f"and its precision ({nb.Precision:.3f}) and recall ({nb.Recall:.3f}) are balanced. "
           if abs(nb.Precision - nb.Recall) < 0.05 else
           f"and its over-confident probabilities favour {'precision' if nb.Precision > nb.Recall else 'recall'} "
           f"(P={nb.Precision:.3f}, R={nb.Recall:.3f}). ")
        + "On the plus side, it is the fastest model to train.")
    o["Support Vector Machine"] = head("Support Vector Machine") + gap_txt("Support Vector Machine") + (
        "The RBF kernel learns a smooth non-linear margin on the standardised features "
        f"(trained on a stratified {meta['best_params']['Support Vector Machine']['train_rows']:,}-row "
        "subsample because kernel SVM training scales as O(n²)). "
        + ("Its high AUC shows a very good ranking of high vs low performers."
           if svm.AUC >= m["AUC"].max() - 0.01 else
           "Its AUC is below the best model, so the kernel margin offers no advantage on this data."))
    runner = m.sort_values(["MCC", "F1", "AUC"], ascending=False).index[1]
    b = m.loc[best]
    o["Overall, Winner for your dataset?"] = (
        f"**{best}** wins with the highest MCC ({b.MCC:.4f}), F1 {b.F1:.4f}, accuracy {b.Accuracy:.4f} "
        f"and AUC {b.AUC:.4f}. The runner-up is {runner} (MCC {m.loc[runner, 'MCC']:.4f}). "
        "MCC is used as the deciding metric because it accounts for all four confusion-matrix cells, "
        "so it is the most reliable single score for a binary classifier.")
    return o


def write_readme(m: pd.DataFrame, meta: dict):
    obs = observations(m, meta)
    rows = "\n".join(
        f"| {k} | {r.Accuracy:.4f} | {r.AUC:.4f} | {r.Precision:.4f} | {r.Recall:.4f} | "
        f"{r.F1:.4f} | {r.MCC:.4f} |" for k, r in m.iterrows())
    obs_rows = "\n".join(f"| {k} | {v} |" for k, v in obs.items())
    nf = meta["numeric_features"] + meta["categorical_features"]
    cb = meta["class_balance"]
    dropped = "; ".join(f"*{k}*: {', '.join(v)}" for k, v in meta["dropped_columns"].items() if v)

    text = f"""# FIFA World Cup 2026 – High-Performer Classification (ML Assignment 2)

## a. Problem statement
Predict whether a player had a **high-performance match** at the FIFA World Cup 2026 from the
player's match statistics (attacking, passing, defensive, disciplinary, goalkeeping and physical
data). This is a **binary classification** problem:

* `{meta['target']} = 1` if the match `{meta['rating_column']}` ≥ **{meta['threshold']:.2f}**
  (the dataset median), otherwise `0`.

Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on the same
data and compared on Accuracy, AUC, Precision, Recall, F1 and MCC. They are deployed in an interactive
Streamlit web app.

## b. Dataset description
* **Source:** Kaggle – [FIFA World Cup 2026 Player Performance Dataset](https://www.kaggle.com/datasets/{meta['dataset']})
  (file `{meta['source_file']}`; synthetically generated by the dataset author).
* **Size:** {meta['n_rows']:,} instances (one row per player per match) × {meta['n_raw_columns']} columns.
  This meets the minimum of 12 features and 500 instances.
* **Feature groups:** player profile (age, height, weight, position, preferred foot, market value),
  match context (stage, result, team/opponent goals), attacking (goals, assists, shots, xG, xA, key
  passes, dribbles), passing (passes, pass accuracy, crosses), defending (tackles, interceptions,
  clearances, blocks, aerial duels, recoveries), discipline (fouls, cards, offsides), goalkeeping
  (saves, save %, clean sheet, goals conceded) and physical (distance, sprint distance, top speed,
  accelerations, stamina).
* **Target:** `{meta['target']}`. Class balance: 1 = {cb.get('1', 0):.2%}, 0 = {cb.get('0', 0):.2%}.
* **Features used ({len(nf)}):** {', '.join(nf)}.
* **Columns removed:** {dropped}.
  These columns were removed because they are the rating itself, rating-derived composite scores,
  whole-tournament aggregates or the Player-of-the-Match flag (all would leak the target), or because
  they are identifiers or high-cardinality names.
* **Pre-processing:** median/mode imputation, StandardScaler for numeric features, One-Hot encoding for
  categorical features. The preprocessing is fitted on the training split only and stored inside each saved pipeline.
* **Split:** stratified 80/20 train/test (random_state = 42) → {meta['train_rows']:,} train /
  {meta['test_rows']:,} test rows. The test split is provided as `test_data.csv`.

## c. GitHub Repository Link
* **GitHub repository:** <PASTE-YOUR-GITHUB-REPO-URL-HERE>
* **Live Streamlit app:** <PASTE-YOUR-STREAMLIT-APP-URL-HERE>

```
project-folder/
│-- app.py                  # Streamlit web application
│-- requirements.txt
│-- README.md
│-- test_data.csv           # 20% hold-out test split (upload this in the app)
│-- model/
│   │-- train_models.py     # source code: data prep, training, evaluation
│   │-- logistic_regression.pkl, decision_tree.pkl, knn.pkl, naive_bayes.pkl, svm.pkl
│   │-- preprocessor.pkl    # fitted preprocessing artifact
│   │-- metadata.json       # features, target threshold, hyper-parameters, versions
│   │-- metrics.csv         # evaluation metrics of all 5 models
```

## d. Models used
Hyper-parameters were tuned with 5-fold stratified GridSearchCV (F1 score) on the training set.
SVM uses an RBF kernel with C = 1 and gamma = 'scale'.

### Comparison Table (evaluation metrics on the hold-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
{rows}

### Observations on the performance of each model

| ML Model Name | Observation about model performance |
|---|---|
{obs_rows}

## How to run
```bash
pip install -r requirements.txt
python model/train_models.py      # downloads data, trains & saves all models, writes this README
streamlit run app.py
```
"""
    with open(os.path.join(ROOT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
