"""
leakage_check.py
-----------------
Find data-leakage in the marketing propensity dataset.

WHAT IT DOES
    A model that scores AUC ~1.0 on a held-out test set is almost always being
    fed a feature that secretly encodes the answer. This script trains a tiny
    model on EACH feature alone and measures how well that single feature
    predicts conversion on unseen data. Any feature whose solo AUC is near 1.0
    is a leak: it should not be in the model.

HOW TO RUN (Colab or local)
    !python leakage_check.py                 # expects data.csv in the folder
    !python leakage_check.py --data mydata.csv

OUTPUT
    - A ranked table (highest solo AUC first).
    - A clear list of FLAGGED features (solo AUC >= --threshold, default 0.95).
    - The full-model AUC, so you can see the inflation for yourself.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Columns that are never model inputs (ID, raw date, the target and its sources)
NON_FEATURES = ["Customer", "Effective To Date", "Response", "Converted", "conversion"]


def build_target(df: pd.DataFrame) -> pd.Series:
    """Conversion = 1 when the customer responded 'Yes' to the offer."""
    if "Response" in df.columns:
        return (df["Response"].astype(str).str.strip().str.lower() == "yes").astype(int)
    if "conversion" in df.columns:
        return df["conversion"].astype(int)
    if "Converted" in df.columns:
        return df["Converted"].astype(int)
    raise ValueError("No 'Response', 'conversion', or 'Converted' column found to build the target.")


def single_feature_model(is_categorical: bool):
    """A small tree that can only use one feature; max_depth caps overfitting."""
    if is_categorical:
        pre = ColumnTransformer(
            [("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0])],
            remainder="drop",
        )
    else:
        pre = ColumnTransformer(
            [("imp", SimpleImputer(strategy="median"), [0])],
            remainder="drop",
        )
    return Pipeline([("pre", pre), ("tree", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv", help="Path to the dataset CSV.")
    ap.add_argument("--threshold", type=float, default=0.95, help="Solo-AUC level that flags a feature as a leak.")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = build_target(df)
    print(f"Rows: {len(df):,}   Conversion rate: {y.mean():.2%}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feature_cols]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    rows = []
    for col in feature_cols:
        is_cat = not pd.api.types.is_numeric_dtype(X[col])
        try:
            model = single_feature_model(is_cat)
            model.fit(X_tr[[col]], y_tr)
            proba = model.predict_proba(X_te[[col]])[:, 1]
            auc = roc_auc_score(y_te, proba)
        except Exception as e:  # noqa: BLE001
            auc = np.nan
        # Highest conversion rate of any single category (purity tell for categoricals)
        purity = np.nan
        if is_cat:
            grp = pd.DataFrame({"v": X_tr[col].astype(str), "y": y_tr.values}).groupby("v")["y"]
            sizes = grp.size()
            means = grp.mean()
            big_enough = means[sizes >= 20]
            purity = float(big_enough.max()) if len(big_enough) else float(means.max())
        rows.append({"feature": col, "type": "cat" if is_cat else "num", "solo_test_AUC": auc, "max_category_rate": purity})

    res = pd.DataFrame(rows).sort_values("solo_test_AUC", ascending=False).reset_index(drop=True)
    res["solo_test_AUC"] = res["solo_test_AUC"].round(4)
    res["max_category_rate"] = res["max_category_rate"].round(3)

    print("\n" + "=" * 70)
    print("SINGLE-FEATURE PREDICTIVE POWER (on a held-out 20% test set)")
    print("=" * 70)
    print(res.to_string(index=False))

    flagged = res[res["solo_test_AUC"] >= args.threshold]["feature"].tolist()
    print("\n" + "-" * 70)
    if flagged:
        print(f"LEAK SUSPECTS (solo AUC >= {args.threshold}): {flagged}")
        print("These features predict the answer almost by themselves — drop them and retrain.")
    else:
        print(f"No single feature exceeds AUC {args.threshold}. If the full model is still")
        print("near-perfect, the leak is in a COMBINATION of features — inspect the top rows.")

    # Show the full-model AUC so the inflation is visible side by side.
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    full_pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ],
        remainder="drop",
    )
    full = Pipeline([("pre", full_pre), ("tree", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE))])
    full.fit(X_tr, y_tr)
    full_auc = roc_auc_score(y_te, full.predict_proba(X_te)[:, 1])
    print("-" * 70)
    print(f"Full-model test AUC (all features): {full_auc:.4f}")
    print("A realistic number for this dataset is roughly 0.70-0.85.")


if __name__ == "__main__":
    main()
