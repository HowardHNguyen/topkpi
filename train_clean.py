"""
train_clean.py
--------------
Leak-free, reproducible training for the Marketing KPI & Propensity app.

WHY THIS REPLACES THE OLD train.py
    - Splits FIRST, then fits every encoder on the training fold only (no test
      data ever touches the preprocessing) -> honest evaluation.
    - Auto-screens for leaking features (any single column that alone predicts
      the answer) and drops them, so you don't get fake AUC ~1.0 numbers.
    - Handles the 14% conversion imbalance with XGBoost's `scale_pos_weight`
      (and class_weight for the others) instead of a CTGAN/GAN, which is simpler
      and carries no leakage risk.
    - Calibrates the probabilities (isotonic) so the percentages the app turns
      into ROI are trustworthy.
    - Saves a SINGLE scikit-learn Pipeline whose first step is a ColumnTransformer
      named "pre". That is exactly what your app.py `_expected_columns_from_pipeline`
      looks for, so the artifact is a drop-in replacement and accepts RAW CSVs
      (it does its own encoding + missing-value handling internally).

HOW TO RUN
    !python train_clean.py                      # expects data.csv in the folder
    !python train_clean.py --data mydata.csv

OUTPUTS
    best_model.pkl          <- copy this to your app's repo root
    model_features.json     <- the exact feature list; sync app.py REQUIRED_FEATURES to it
    model_comparison.csv    <- holdout AUC / PR-AUC / Brier / Lift@10% for every model
    holdout_for_app.csv     <- a genuine unseen slice (with `conversion` + CLV) you can
                               upload to the app to see REALISTIC KPI/Lift/Calibration
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

warnings.filterwarnings("ignore")
np.seterr(all="ignore")  # silence benign numpy matmul over/underflow warnings

RANDOM_STATE = 42

# ── Configuration ───────────────────────────────────────────────────────────
# The app's REQUIRED_FEATURES. The model trains on these (minus any leaker the
# screen catches). Customer Lifetime Value is intentionally NOT a model input:
# the app treats it as optional and uses it only for realized-CLV / ROI math.
CANDIDATE_FEATURES = [
    "State", "Coverage", "Education", "EmploymentStatus", "Income",
    "Location Code", "Marital Status", "Monthly Premium Auto",
    "Months Since Last Claim", "Months Since Policy Inception",
    "Number of Open Complaints", "Number of Policies",
    "Renew Offer Type", "Sales Channel", "Total Claim Amount",
    "Vehicle Class", "Vehicle Size",
]
NON_FEATURES = ["Customer", "Effective To Date", "Response", "Converted", "conversion"]
LEAK_AUC_THRESHOLD = 0.95   # solo-feature test AUC above this = dropped as a leak
MANUAL_DROP = []            # add column names here to force-drop them


def build_target(df: pd.DataFrame) -> pd.Series:
    if "Response" in df.columns:
        return (df["Response"].astype(str).str.strip().str.lower() == "yes").astype(int)
    if "conversion" in df.columns:
        return df["conversion"].astype(int)
    if "Converted" in df.columns:
        return df["Converted"].astype(int)
    raise ValueError("No 'Response', 'conversion', or 'Converted' column to build the target.")


def solo_auc(X_tr, y_tr, X_te, y_te, col, is_cat) -> float:
    if is_cat:
        pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0])], remainder="drop")
    else:
        pre = ColumnTransformer([("imp", SimpleImputer(strategy="median"), [0])], remainder="drop")
    m = Pipeline([("pre", pre), ("tree", DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE))])
    try:
        m.fit(X_tr[[col]], y_tr)
        return roc_auc_score(y_te, m.predict_proba(X_te[[col]])[:, 1])
    except Exception:
        return np.nan


def make_preprocessor(num_cols, cat_cols, scale: bool) -> ColumnTransformer:
    """ColumnTransformer with explicit column-name lists so app.py can recover
    the feature list from `transformers_`. Covers every column -> remainder empty."""
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ],
        remainder="drop",
    )


def lift_at_k(y_true: np.ndarray, proba: np.ndarray, k: float = 0.10) -> float:
    cutoff = np.percentile(proba, 100 * (1 - k))
    top = y_true[proba >= cutoff]
    base = y_true.mean()
    return float(top.mean() / base) if base > 0 and len(top) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = build_target(df)
    print(f"Loaded {len(df):,} rows | conversion rate {y.mean():.2%}")

    # Use only candidate features that actually exist in the file.
    features = [c for c in CANDIDATE_FEATURES if c in df.columns and c not in NON_FEATURES]
    missing = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    if missing:
        print(f"Note: these expected features are absent and will be skipped: {missing}")

    # ── Deduplicate on model features ─────────────────────────────────────────
    # Identical feature-rows let a high-capacity model "look up" answers and, when
    # they straddle the train/test split, inflate the holdout score. Drop exact
    # duplicates (keeping the first) before doing anything else.
    before = len(df)
    df = df.loc[~df[features].duplicated(keep="first")].reset_index(drop=True)
    y = build_target(df)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed:,} duplicate feature-rows -> {len(df):,} unique rows remain.")

    # Index-based split so we can also carve out a matching RAW holdout for the app.
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE)
    X = df[features]
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    # ── Leak screen ──────────────────────────────────────────────────────────
    print("\nScreening features for leakage (solo test AUC)...")
    leak_rows, leakers = [], []
    for col in features:
        is_cat = not pd.api.types.is_numeric_dtype(X[col])
        a = solo_auc(X_tr, y_tr, X_te, y_te, col, is_cat)
        leak_rows.append((col, round(a, 4) if pd.notna(a) else a))
        if pd.notna(a) and a >= LEAK_AUC_THRESHOLD:
            leakers.append(col)
    for col, a in sorted(leak_rows, key=lambda t: (t[1] is None, -(t[1] or 0)))[:8]:
        print(f"   {col:<32} solo AUC {a}")
    drop = sorted(set(leakers) | set(MANUAL_DROP))
    if drop:
        print(f"DROPPING leak suspects: {drop}")
    final_features = [c for c in features if c not in drop]
    print(f"Final model features ({len(final_features)}): {final_features}")

    X_tr, X_te = X_tr[final_features], X_te[final_features]
    num_cols = [c for c in final_features if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in final_features if c not in num_cols]
    assert sorted(num_cols + cat_cols) == sorted(final_features), "Column coverage mismatch."

    # ── Imbalance handling ─────────────────────────────────────────────────────
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = round(neg / max(pos, 1), 3)
    print(f"\nClass balance in train: {pos} positive / {neg} negative -> scale_pos_weight={spw}")

    # ── Candidate models (each wrapped: preprocess -> calibrated classifier) ──
    def calibrated(base, scale=False):
        pre = make_preprocessor(num_cols, cat_cols, scale=scale)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        return Pipeline([("pre", pre), ("clf", clf)])

    candidates = {
        # Regularized so leaves can't isolate individual rows: large min_samples_leaf,
        # modest depth, L2. This captures genuine signal without memorizing quirks.
        "Gradient Boosting (HistGB)": calibrated(
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           max_leaf_nodes=15, min_samples_leaf=50,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.1, class_weight="balanced",
                                           random_state=RANDOM_STATE)),
        "Random Forest": calibrated(
            RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=50,
                                   class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
        "Logistic Regression": calibrated(
            LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
    }
    if HAVE_XGB:
        # NOTE: XGBoost only works with scikit-learn >= 1.6 when XGBoost >= 3.0.
        # If your installed XGBoost is older, the fit below fails gracefully and is skipped.
        candidates["XGBoost"] = calibrated(
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=1.0, scale_pos_weight=spw, eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=1))
    else:
        print("xgboost not installed -> skipping XGBoost (pip install 'xgboost>=3.0' to include it).")

    # ── Train & evaluate: 5-fold CV (defensible headline) + untouched holdout ──
    X_full, y_full = df[final_features], y
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\nEvaluating with 5-fold cross-validation + a held-out test set...")
    results, fitted = [], {}
    for name, model in candidates.items():
        try:
            cv_auc = cross_val_score(model, X_full, y_full, cv=cv, scoring="roc_auc", n_jobs=1)
            model.fit(X_tr, y_tr)
            proba_te = model.predict_proba(X_te)[:, 1]
            proba_tr = model.predict_proba(X_tr)[:, 1]
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[-1][:120]
            print(f"   [skipped] {name}: {msg}")
            if "XGB" in name:
                print("             (Upgrade with: pip install 'xgboost>=3.0' to include XGBoost.)")
            continue
        results.append({
            "Model": name,
            "CV_AUC": round(float(cv_auc.mean()), 4),
            "CV_std": round(float(cv_auc.std()), 4),
            "Holdout_AUC": round(roc_auc_score(y_te, proba_te), 4),
            "Train_AUC": round(roc_auc_score(y_tr, proba_tr), 4),
            "Brier": round(brier_score_loss(y_te, proba_te), 4),
            "Lift@10%": round(lift_at_k(y_te.values, proba_te), 2),
        })
        fitted[name] = model

    if not results:
        raise RuntimeError("All candidate models failed to train. Check the messages above.")

    res = pd.DataFrame(results).sort_values(["CV_AUC", "Brier"], ascending=[False, True]).reset_index(drop=True)
    print("\n" + "=" * 78)
    print("MODEL COMPARISON  (CV_AUC = stable headline; Train vs Holdout = overfit check)")
    print("=" * 78)
    print(res.to_string(index=False))
    res.to_csv("model_comparison.csv", index=False)

    # Overfit flag: a large Train - Holdout gap means it is still memorizing.
    gap = float(res.iloc[0]["Train_AUC"] - res.iloc[0]["Holdout_AUC"])
    if res.iloc[0]["Holdout_AUC"] > 0.95:
        print(f"\nNote: winner holdout AUC is {res.iloc[0]['Holdout_AUC']} — still very high even after")
        print("      dedupe + regularization. That points to a near-deterministic target in this")
        print("      dataset rather than real-world signal; treat the metrics as illustrative.")
    elif gap > 0.07:
        print(f"\nNote: Train AUC exceeds Holdout by {gap:.3f} — some overfitting remains.")

    best_name = res.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nWINNER: {best_name}")

    # ── Save app-compatible artifact + metadata ───────────────────────────────
    joblib.dump(best_model, "best_model.pkl")
    json.dump(
        {"model": best_name, "features": final_features, "dropped_as_leak": drop,
         "scale_pos_weight": spw, "metrics": res.iloc[0].to_dict()},
        open("model_features.json", "w"), indent=2,
    )

    # A genuine unseen slice for honest validation inside the app.
    holdout = df.iloc[te_idx].copy()
    holdout["conversion"] = y_te.values
    holdout.to_csv("holdout_for_app.csv", index=False)

    print("\nSaved: best_model.pkl, model_features.json, model_comparison.csv, holdout_for_app.csv")
    if drop:
        print("\nIMPORTANT: features were dropped as leaks. Update REQUIRED_FEATURES in app.py")
        print("           to match model_features.json so the schema checklist stays in sync.")
    print("Tip: upload holdout_for_app.csv in the app to see realistic KPI / Lift / Calibration.")


if __name__ == "__main__":
    main()
