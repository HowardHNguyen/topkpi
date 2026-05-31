# app.py
# Advanced Marketing KPI Performance with Data Science
# Streamlit app (root-level). Loads best_model.pkl (joblib).
# Safe Plotly imports + guards; no sklearn shim code.
#
# Upgrades vs previous version:
#   - Model cached with st.cache_resource (correct cache for fitted pipelines)
#   - Single cached scoring pass reused across all sections (no triple-scoring)
#   - Robust positive-class detection in predict_proba
#   - pandas 2.x-clean KPI grouping (observed=True, no fragile .apply/.get)
#   - Real cumulative-gain curve added to "Lift & Gain"
#   - AUC / Average Precision / Brier tiles shown when ground truth is present
#   - Profit-vs-threshold optimizer recommends the ROI-maximizing operating point
#   - Download top-N prospects; cleaned dead imports / stale comments

import os
import sys
import time
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st

# ───────────────────────────────────────────────────────────────────────────────
# Optional dependencies (app must not crash if a viz/metric lib is missing)
# ───────────────────────────────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    px, go, PLOTLY_OK = None, None, False

try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    joblib = None
    HAVE_JOBLIB = False
import pickle

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    SKMETRICS_OK = True
except Exception:
    SKMETRICS_OK = False


def load_any_pickle(path: str):
    """Load a model artifact via joblib, falling back to stdlib pickle."""
    if HAVE_JOBLIB:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# === Schema requirements ===
REQUIRED_FEATURES = [
    "State", "Coverage", "Education", "EmploymentStatus", "Income",
    "Location Code", "Marital Status", "Monthly Premium Auto",
    "Months Since Last Claim", "Months Since Policy Inception",
    "Number of Open Complaints", "Number of Policies",
    "Renew Offer Type", "Sales Channel", "Total Claim Amount",
    "Vehicle Class", "Vehicle Size",
]
OPTIONAL_FEATURES = [
    "conversion",              # enables back-tested KPIs, Lift, Gain, Calibration
    "Customer Lifetime Value",  # enables realized CLV; proxy used if missing
]

# ───────────────────────────────────────────────────────────────────────────────
# App config
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing KPI & Propensity Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Minimal + safe: hide only right-side actions and bottom-right watermark ---
st.markdown(
    """
<style>
header { visibility: visible !important; }
div[data-testid="stDecoration"] { display: block !important; visibility: visible !important; }
div[data-testid="stSidebar"] { visibility: visible !important; display: block !important; }
div[data-testid="stToolbarActions"] { display: none !important; }
div[data-testid="stToolbar"] { display: block !important; visibility: visible !important; }
.stAppBottomRightButtons, .stAppDeployButton { display: none !important; }
#MainMenu { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

with st.expander("About This Project", expanded=False):
    st.markdown(
        """
        ### 💡 About This Project
        This app and its accompanying article — **Advanced Marketing KPI Performance with Data Science** — bridge the gap
        between traditional marketing metrics and modern AI intelligence.

        In most marketing teams, KPIs such as **Conversion Rate (CR)**, **Customer Lifetime Value (CLV)**,
        **Cost per Acquisition (CPA)**, and **Return on Investment (ROI)** are analyzed separately. This project integrates
        them within a **predictive data-science framework** that reveals how they interact, enabling data-driven
        decision-making instead of static reporting.

        ### 🎯 Purpose
        Empower marketing leaders, analysts, and growth teams to **move from descriptive to prescriptive insights** —
        forecasting conversions, simulating ROI, and recommending optimal channel strategies.

        ### ⚙️ How It Was Built
        - **Dataset:** Customer and policy data (demographics, claims, sales channels, offers).
        - **Modeling Stack:** `scikit-learn`, `LightGBM`, `XGBoost`, and a calibrated ensemble pipeline for **propensity modeling**.
        - **Metrics:** CR, CLV, CPA, and ROI dynamically computed for every uploaded dataset.
        - **Validation:** Stratified CV, grouped CV by customer, and permutation AUC to detect overfitting.
        - **Visualization:** Interactive **Plotly** dashboards with “How to Read This Section” notes.
        - **Deployment:** Streamlit + GitHub + Python 3.12 (`scikit-learn 1.6.1`, `lightgbm 4.5.0`, `joblib 1.4.2`).
        """
    )

DEFAULT_COST_MAP = {"Web": 40, "Call Center": 70, "Branch": 90, "Agent": 120}
TARGET_COL = "conversion"
CLV_COL = "Customer Lifetime Value"  # optional; proxy used if missing

# ───────────────────────────────────────────────────────────────────────────────
# Diagnostics (sidebar expander)
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("Environment diagnostics", expanded=False):
    st.write("Python:", sys.version)
    st.write("Plotly installed:", importlib.util.find_spec("plotly") is not None)

    def _ver(name):
        try:
            m = __import__(name)
            return getattr(m, "__version__", "unknown")
        except Exception:
            return "missing"

    st.write(
        {
            "plotly": _ver("plotly"),
            "streamlit": _ver("streamlit"),
            "numpy": _ver("numpy"),
            "pandas": _ver("pandas"),
            "scikit_learn": _ver("sklearn"),
            "lightgbm": _ver("lightgbm"),
            "xgboost": _ver("xgboost"),
            "joblib": _ver("joblib"),
        }
    )

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _expected_columns_from_pipeline(pipe):
    """Recover original feature names the ColumnTransformer expects."""
    try:
        pre = pipe.named_steps.get("pre") or pipe.named_steps.get("pre3")
        cols = []
        for _, _, cols_sel in pre.transformers_:
            if cols_sel == "drop":
                continue
            if isinstance(cols_sel, list):
                cols.extend(cols_sel)
            elif isinstance(cols_sel, (tuple, np.ndarray, pd.Index)):
                cols.extend(list(cols_sel))
        return list(dict.fromkeys(cols))  # unique, keep order
    except Exception:
        return None


def ensure_columns_for_pipeline(df: pd.DataFrame, expected: list) -> pd.DataFrame:
    """Ensure df has exactly the columns the model expects (add missing as NaN; keep order)."""
    if expected is None:
        return df
    out = df.copy()
    for c in expected:
        if c not in out.columns:
            out[c] = np.nan
    return out[expected]


def _positive_class_index(pipe) -> int:
    """Find the column index of the positive class (label 1) in predict_proba output."""
    classes = getattr(pipe, "classes_", None)
    if classes is not None:
        classes = list(classes)
        for target in (1, "1", True, "Yes", "yes"):
            if target in classes:
                return classes.index(target)
    return -1  # default: last column


def predict_propensity(pipe, X: pd.DataFrame) -> np.ndarray:
    """Robust positive-class probability extraction across binary/edge-case models."""
    proba = np.asarray(pipe.predict_proba(X))
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, _positive_class_index(pipe)]


def proxy_clv(df: pd.DataFrame) -> pd.Series:
    """Proxy CLV if the CLV column is missing (12 months × premium × 35% margin)."""
    monthly = df.get("Monthly Premium Auto", pd.Series(np.nan, index=df.index)).astype(float)
    months, margin = 12.0, 0.35
    prox = months * monthly * margin
    if prox.isna().all():
        prox = pd.Series(12.0 * 100.0 * 0.35, index=df.index)  # fallback $420
    return prox


def _group_kpis(d: pd.DataFrame, group_col: str, clv_col: str) -> pd.DataFrame:
    """Per-group CR / Acquired / CPA / CLV realized / ROI without fragile groupby.apply."""
    work = d.copy()
    work["_clv"] = work[clv_col].clip(lower=0) if clv_col in work.columns else proxy_clv(work)
    work["_target"] = work[TARGET_COL] if TARGET_COL in work.columns else 0
    work["_cost_converted"] = np.where(work["_target"] == 1, work["est_acquisition_cost"], 0.0)
    work["_clv_realized"] = work["_clv"] * work["_target"]

    g = work.groupby(group_col, observed=True).agg(
        conversion_rate=("_target", "mean"),
        acquired=("_target", "sum"),
        total_cost=("_cost_converted", "sum"),
        clv_realized=("_clv_realized", "sum"),
    )
    g["acquired"] = g["acquired"].astype(int)
    g["cpa"] = g["total_cost"] / g["acquired"].replace(0, np.nan)
    g["roi"] = (g["clv_realized"] - g["total_cost"]) / g["total_cost"].replace(0, np.nan)
    return g.drop(columns="total_cost").reset_index()


def compute_kpis(
    df: pd.DataFrame,
    cost_map: dict = None,
    channel_col: str = "Sales Channel",
    offer_col: str = "Renew Offer Type",
    clv_col: str = CLV_COL,
) -> dict:
    d = df.copy()
    cost_map = cost_map or DEFAULT_COST_MAP

    if channel_col in d.columns:
        d["est_acquisition_cost"] = d[channel_col].map(cost_map).fillna(np.median(list(cost_map.values())))
    else:
        d["est_acquisition_cost"] = np.median(list(cost_map.values()))

    clv = d[clv_col].clip(lower=0) if clv_col in d.columns else proxy_clv(d)

    conv_rate = float(d[TARGET_COL].mean()) if TARGET_COL in d.columns else np.nan
    acquired = int(d[TARGET_COL].sum()) if TARGET_COL in d.columns else np.nan

    total_cost = d.loc[d.get(TARGET_COL, pd.Series(False)) == 1, "est_acquisition_cost"].sum()
    clv_realized = float((clv * d.get(TARGET_COL, 0)).sum())
    cpa = float(total_cost / max(acquired, 1)) if acquired and acquired > 0 else np.nan
    roi = float((clv_realized - total_cost) / total_cost) if total_cost > 0 else np.nan

    out = {
        "overall": {
            "conversion_rate": conv_rate,
            "acquired": acquired,
            "cpa": cpa,
            "clv_realized": clv_realized,
            "roi": roi,
        }
    }

    if channel_col in d.columns:
        out["by_channel"] = _group_kpis(d, channel_col, clv_col)
    if offer_col in d.columns:
        out["by_offer"] = _group_kpis(d, offer_col, clv_col)
    return out


def lift_table(y_true: pd.Series, y_pred: pd.Series, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.values, "p": y_pred.values})
    df["bucket"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    agg = (
        df.groupby("bucket", observed=True)
        .agg(n=("y", "size"), positives=("y", "sum"), avg_p=("p", "mean"))
        .sort_values("avg_p", ascending=False)
        .reset_index(drop=True)
    )
    agg["rate"] = agg["positives"] / agg["n"].replace(0, np.nan)
    agg["cum_positives"] = agg["positives"].cumsum()
    agg["cum_n"] = agg["n"].cumsum()
    agg["cum_rate"] = agg["cum_positives"] / agg["cum_n"].replace(0, np.nan)
    base = df["y"].mean() if df["y"].notna().any() else np.nan
    agg["lift"] = agg["rate"] / base if base and base > 0 else np.nan
    agg["cum_lift"] = agg["cum_rate"] / base if base and base > 0 else np.nan
    # Cumulative gain = % of all positives captured by the top-k ranked records
    total_pos = df["y"].sum()
    agg["cum_gain"] = agg["cum_positives"] / total_pos if total_pos > 0 else np.nan
    agg["pct_population"] = agg["cum_n"] / len(df)
    return agg


def fmt_pct(x):
    return "-" if pd.isna(x) else f"{100*x:.2f}%"


def fmt_money(x):
    return "-" if pd.isna(x) else f"${x:,.0f}"


def render_schema_checklist(df):
    with st.sidebar.expander("Schema checklist", expanded=False):
        if df is None:
            st.info("Upload a CSV to validate the schema. Use `sample_data.csv` as a template.")
            st.caption("Required features drive scoring; optional fields enable realized CLV / actual-KPI back-tests.")
            return
        cols = set(df.columns)
        st.markdown("**Required features**")
        for f in REQUIRED_FEATURES:
            st.markdown(("🟢 " if f in cols else "🔴 ") + f)
        st.markdown("**Optional features**")
        for f in OPTIONAL_FEATURES:
            st.markdown(("🟢 " if f in cols else "⚪ ") + f)
        st.caption(
            "- Categorical values not seen in training are handled safely.\n"
            "- If `conversion` is missing, KPIs are **estimated** from scores until actuals arrive.\n"
            "- If `Customer Lifetime Value` is missing, a **conservative proxy** keeps ROI working."
        )


# ───────────────────────────────────────────────────────────────────────────────
# Cached I/O
# ───────────────────────────────────────────────────────────────────────────────
MODEL_CANDIDATES = ["best_model.pkl"]  # ensure this file is in the repo root


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Fitted pipelines are resources, not data → cache_resource (no re-pickling)."""
    return load_any_pickle(path)


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner="Scoring…")
def score_dataframe(df: pd.DataFrame, _pipe, expected_cols) -> np.ndarray:
    """Score once and reuse across sections. `_pipe` is unhashable, so it is excluded
    from the cache key (leading underscore); df + expected_cols form the key."""
    df_for_model = ensure_columns_for_pipeline(df, expected_cols)
    return predict_propensity(_pipe, df_for_model)


pipe = None
model_info = ""
for p in MODEL_CANDIDATES:
    if os.path.exists(p):
        try:
            pipe = load_model(p)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
            model_info = f"Loaded: {p} (modified {ts})"
            break
        except Exception as e:
            st.sidebar.error(f"Failed loading {p}: {e}")

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.title("MaxAIS • KPI + Propensity")
st.sidebar.caption("Advanced Marketing KPI Performance — Conversion • CLV • CPA • ROI • Propensity")

with st.sidebar.expander("Cost per Acquisition (override)", expanded=False):
    web = st.number_input("Web", value=int(DEFAULT_COST_MAP["Web"]), min_value=0, step=5)
    cc = st.number_input("Call Center", value=int(DEFAULT_COST_MAP["Call Center"]), min_value=0, step=5)
    br = st.number_input("Branch", value=int(DEFAULT_COST_MAP["Branch"]), min_value=0, step=5)
    ag = st.number_input("Agent", value=int(DEFAULT_COST_MAP["Agent"]), min_value=0, step=5)
user_cost_map = {"Web": web, "Call Center": cc, "Branch": br, "Agent": ag}

section = st.sidebar.radio(
    "Navigate",
    ["📥 Data", "📊 KPIs", "🤖 Propensity", "📈 Lift & Gain", "🧪 Calibration"],
    index=1,
)

if pipe is not None:
    st.sidebar.success(model_info)
else:
    st.sidebar.error(f"No model loaded. Ensure {MODEL_CANDIDATES[0]} is in the repo root.")

# ───────────────────────────────────────────────────────────────────────────────
# Data input
# ───────────────────────────────────────────────────────────────────────────────
st.title("Marketing KPI & Propensity Intelligence by Howard Nguyen")
st.write(
    "**Conversion • CLV • CPA • ROI • Propensity** — actionable, executive-ready analytics with calibrated predictions."
)

uploaded = st.file_uploader("Upload CSV (same schema as training)", type=["csv"], accept_multiple_files=False)

df = None
if uploaded is not None:
    try:
        df = read_csv_cached(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
elif os.path.exists("sample_data.csv"):
    try:
        df = pd.read_csv("sample_data.csv")
        st.info("Loaded sample_data.csv from repo root. Upload your CSV to replace it.")
    except Exception as e:
        st.error(f"Could not read sample_data.csv: {e}")
else:
    st.warning("Upload a CSV to begin (or include sample_data.csv in the repo root).")

if df is not None:
    st.write("**Preview**")
    st.dataframe(df.head(25), use_container_width=True)

has_target = df is not None and (TARGET_COL in df.columns)
render_schema_checklist(df)

# Score once here; every section below reuses `scores` / `df_scored`.
scores = None
df_scored = None
expected_cols = _expected_columns_from_pipeline(pipe) if pipe is not None else None
if df is not None and pipe is not None:
    try:
        scores = score_dataframe(df, pipe, expected_cols)
        df_scored = df.copy()
        df_scored["propensity"] = scores
    except Exception as e:
        st.error(f"Scoring failed: {e}")

# ───────────────────────────────────────────────────────────────────────────────
# KPIs
# ───────────────────────────────────────────────────────────────────────────────
if section == "📊 KPIs":
    st.subheader("KPI Overview")
    with st.expander("How to read this section", expanded=False):
        st.markdown(
            """
            **Goal:** See where profit is created (by channel/offer) and whether budget is efficient.

            **Tiles**
            - **Conversion Rate** – % of rows with `conversion=1`.
            - **Acquired** – number of converted customers.
            - **CPA** – average acquisition cost for **converted** customers (uses sidebar CPAs).
            - **CLV Realized** – sum of realized value from converters (uses CLV column or proxy).
            - **ROI** – `(CLV Realized − Spend) / Spend`.

            **Good to know**
            - If CLV is missing, the app uses a **conservative CLV proxy** so ROI still works.
            - Change the **CPA sliders** (left) to simulate media price changes; KPIs update instantly.
            """
        )

    if df is None:
        st.warning("Upload data to compute KPIs.")
    elif not has_target:
        st.info(
            "No ground-truth conversion column found. Switch to **Propensity** to score first; "
            "you’ll see **Estimated KPIs** there."
        )
    else:
        kpis = compute_kpis(df, cost_map=user_cost_map)
        overall = kpis["overall"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Conversion Rate", fmt_pct(overall["conversion_rate"]))
        c2.metric("Acquired", overall["acquired"])
        c3.metric("CPA", fmt_money(overall["cpa"]))
        c4.metric("CLV Realized", fmt_money(overall["clv_realized"]))
        c5.metric("ROI", fmt_pct(overall["roi"]))

        if "by_channel" in kpis and len(kpis["by_channel"]):
            by_ch = kpis["by_channel"]
            st.markdown("#### By Sales Channel")
            st.dataframe(by_ch, use_container_width=True)
            if not PLOTLY_OK:
                st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
            else:
                fig = px.bar(
                    by_ch, x="Sales Channel", y="conversion_rate",
                    title="Conversion Rate by Channel",
                    text=by_ch["conversion_rate"].map(lambda v: f"{100*v:.1f}%"),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    "**Interpretation:** Use this to **shift spend** — scale channels with higher conversion and "
                    "acceptable CPA, and combine with **ROI by Offer** to confirm profitability, not just efficiency."
                )

        if "by_offer" in kpis and len(kpis["by_offer"]):
            by_offer = kpis["by_offer"]
            st.markdown("#### By Offer Type")
            st.dataframe(by_offer, use_container_width=True)
            if not PLOTLY_OK:
                st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
            else:
                fig2 = px.bar(
                    by_offer, x="Renew Offer Type", y="roi",
                    title="ROI by Offer Type",
                    text=by_offer["roi"].map(lambda v: f"{100*v:.1f}%"),
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown(
                    "**Interpretation:** **Pause** offers with near-zero ROI, **scale** offers with strong ROI **and** "
                    "sufficient volume, and re-check ROI against your hurdle whenever CPA rises (sidebar)."
                )

# ───────────────────────────────────────────────────────────────────────────────
# Propensity
# ───────────────────────────────────────────────────────────────────────────────
elif section == "🤖 Propensity":
    st.subheader("Propensity Scoring")
    with st.expander("How to read this section", expanded=False):
        st.markdown(
            """
            **Goal:** Rank customers by likelihood to convert and decide *how deep* to target.

            - **Top prospects** – highest probability first (outreach lists or routing).
            - **Propensity Distribution** – wider right tail = more “easy wins.”
            - **Threshold slider** – pick the **operating point** (e.g., top 20–30%).
            - **Profit curve** – the app recommends the threshold that **maximizes ROI**.
            """
        )

    if df is None:
        st.warning("Upload data to score.")
    elif pipe is None:
        st.warning(f"Model not loaded. Put {MODEL_CANDIDATES[0]} in repo root.")
    elif df_scored is None:
        st.warning("Scoring unavailable — see the error above.")
    else:
        st.write("**Top prospects** (highest probability first):")
        top_view = df_scored.sort_values("propensity", ascending=False)
        st.dataframe(top_view.head(25), use_container_width=True)

        colA, colB = st.columns([2, 1])
        with colA:
            if not PLOTLY_OK:
                st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
            else:
                figp = px.histogram(df_scored, x="propensity", nbins=30, title="Propensity Distribution")
                st.plotly_chart(figp, use_container_width=True)
        with colB:
            thr = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            est_converted = int((df_scored["propensity"] >= thr).sum())
            st.metric("Targeted @ threshold", est_converted)
            st.caption("**Threshold** — sets how deep you go. Lower = more volume, lower precision.")

        # Profit / ROI vs threshold optimizer ------------------------------------
        st.markdown("#### Profit & ROI by threshold")
        med_cost = float(np.median(list(user_cost_map.values())))
        if CLV_COL in df_scored.columns:
            clv_series = df_scored[CLV_COL].clip(lower=0).fillna(proxy_clv(df_scored))
        else:
            clv_series = proxy_clv(df_scored)

        grid = np.round(np.arange(0.0, 1.001, 0.02), 3)
        rows = []
        for t in grid:
            mask = df_scored["propensity"] >= t
            n_target = int(mask.sum())
            if n_target == 0:
                rows.append({"threshold": t, "targeted": 0, "value": 0.0, "spend": 0.0, "profit": 0.0, "roi": np.nan})
                continue
            if has_target:
                # Realized: value only from true converters inside the targeted set
                conv_mask = mask & (df_scored[TARGET_COL] == 1)
                value = float((clv_series[conv_mask]).sum())
            else:
                # Expected: probability-weighted value across the targeted set
                value = float((df_scored.loc[mask, "propensity"] * clv_series[mask]).sum())
            spend = n_target * med_cost
            profit = value - spend
            roi = (profit / spend) if spend > 0 else np.nan
            rows.append({"threshold": t, "targeted": n_target, "value": value, "spend": spend, "profit": profit, "roi": roi})
        prof_df = pd.DataFrame(rows)

        best = prof_df.loc[prof_df["profit"].idxmax()] if prof_df["profit"].notna().any() else None
        if best is not None:
            b1, b2, b3 = st.columns(3)
            b1.metric("Recommended threshold", f"{best['threshold']:.2f}")
            b2.metric("Profit at recommendation", fmt_money(best["profit"]))
            b3.metric("ROI at recommendation", fmt_pct(best["roi"]))
            st.caption(
                ("Profit uses **realized** converter value (ground truth present)."
                 if has_target else
                 "No ground truth — profit uses **expected** (probability-weighted) value.")
                + f" Spend assumes median CPA ${med_cost:,.0f}/target."
            )

        if PLOTLY_OK and len(prof_df):
            figpf = go.Figure()
            figpf.add_trace(go.Scatter(x=prof_df["threshold"], y=prof_df["profit"], name="Profit", mode="lines"))
            if best is not None:
                figpf.add_vline(x=float(best["threshold"]), line_dash="dash",
                                annotation_text="recommended", annotation_position="top")
            figpf.update_layout(title="Profit vs Threshold", xaxis_title="Threshold", yaxis_title="Profit ($)")
            st.plotly_chart(figpf, use_container_width=True)

        # Estimated KPIs when no ground truth ------------------------------------
        if not has_target:
            dtmp = df_scored.copy()
            dtmp[TARGET_COL] = (dtmp["propensity"] >= thr).astype(int)
            overall = compute_kpis(dtmp, cost_map=user_cost_map)["overall"]
            st.markdown("#### Estimated KPIs (propensity ≥ threshold)")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Conversion Rate", fmt_pct(overall["conversion_rate"]))
            c2.metric("Acquired", overall["acquired"])
            c3.metric("CPA", fmt_money(overall["cpa"]))
            c4.metric("CLV Realized", fmt_money(overall["clv_realized"]))
            c5.metric("ROI", fmt_pct(overall["roi"]))
            st.caption("Estimated KPIs use model scores (no ground truth). Use them to **size campaigns** pre-launch.")

        # Downloads --------------------------------------------------------------
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download scored CSV (all)",
                data=df_scored.to_csv(index=False).encode("utf-8"),
                file_name="scored_output.csv",
                mime="text/csv",
            )
        with d2:
            top_n = st.number_input("Top-N prospects", min_value=10, max_value=int(len(df_scored)),
                                    value=min(500, len(df_scored)), step=50)
            st.download_button(
                f"Download top {int(top_n)} prospects",
                data=top_view.head(int(top_n)).to_csv(index=False).encode("utf-8"),
                file_name=f"top_{int(top_n)}_prospects.csv",
                mime="text/csv",
            )

# ───────────────────────────────────────────────────────────────────────────────
# Lift & Gain
# ───────────────────────────────────────────────────────────────────────────────
elif section == "📈 Lift & Gain":
    st.subheader("Lift & Cumulative Gain")
    with st.expander("How to read this section", expanded=False):
        st.markdown(
            """
            **Goal:** Prove the model’s business value and decide how many deciles to target.

            - **Lift** – conversion rate in a decile vs the overall average (e.g., 6× in the top decile).
            - **Cumulative Gain** – % of all converters captured as you target more of the ranked list.

            **Rules of thumb**
            - Top-decile lift **> 3×** is strong; **> 5×** is excellent.
            - A gain curve that rises steeply then flattens means most value sits in the top deciles.
            """
        )

    if df is None:
        st.warning("Upload data to compute lift.")
    elif pipe is None:
        st.warning("Model not loaded.")
    elif not has_target:
        st.info("Lift & Gain require a ground-truth `conversion` column.")
    elif df_scored is None:
        st.warning("Scoring unavailable — see the error above.")
    else:
        lift = lift_table(df_scored[TARGET_COL], df_scored["propensity"], bins=10)
        st.dataframe(lift, use_container_width=True)

        if not PLOTLY_OK:
            st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
        else:
            fig_lift = px.line(lift, y="lift", title="Lift by Decile", markers=True)
            fig_lift.update_layout(xaxis_title="Decile (high→low propensity)", yaxis_title="Lift")
            st.plotly_chart(fig_lift, use_container_width=True)

            fig_cum = px.line(lift, y="cum_lift", title="Cumulative Lift", markers=True)
            fig_cum.update_layout(xaxis_title="Decile (cumulative)", yaxis_title="Cumulative Lift")
            st.plotly_chart(fig_cum, use_container_width=True)

            # Real cumulative-gain curve with the random baseline
            fig_gain = go.Figure()
            fig_gain.add_trace(go.Scatter(
                x=[0] + lift["pct_population"].tolist(),
                y=[0] + lift["cum_gain"].tolist(),
                mode="lines+markers", name="Model",
            ))
            fig_gain.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random",
                line=dict(dash="dash"),
            ))
            fig_gain.update_layout(
                title="Cumulative Gain Curve",
                xaxis_title="% of population targeted (ranked)",
                yaxis_title="% of converters captured",
            )
            st.plotly_chart(fig_gain, use_container_width=True)
            st.markdown(
                "**Interpretation:** the further the model curve bows above the diagonal, the more converters you "
                "capture for a given amount of outreach. A steep early rise means you can target shallow and still hit goals."
            )

# ───────────────────────────────────────────────────────────────────────────────
# Calibration
# ───────────────────────────────────────────────────────────────────────────────
elif section == "🧪 Calibration":
    st.subheader("Calibration Check (Binning)")
    with st.expander("How to read this section", expanded=False):
        st.markdown(
            """
            **Goal:** Verify that predicted probabilities match real-world outcomes.

            - Line close to the dashed **diagonal** = **well-calibrated**.
            - **Above** the diagonal: model is **under**-predicting; **below**: **over**-predicting.

            **Why it matters:** good calibration makes **ROI forecasts** (score × CLV) trustworthy. If miscalibrated,
            re-calibrate (isotonic / Platt) or retrain with recent data.
            """
        )

    if df is None:
        st.warning("Upload data to check calibration.")
    elif pipe is None:
        st.warning("Model not loaded.")
    elif not has_target:
        st.info("Calibration requires a ground-truth `conversion` column.")
    elif df_scored is None:
        st.warning("Scoring unavailable — see the error above.")
    else:
        y_true = df_scored[TARGET_COL].astype(float)
        y_pred = df_scored["propensity"].astype(float)

        # Model-quality tiles
        if SKMETRICS_OK and y_true.nunique() > 1:
            m1, m2, m3 = st.columns(3)
            try:
                m1.metric("ROC AUC", f"{roc_auc_score(y_true, y_pred):.3f}")
            except Exception:
                m1.metric("ROC AUC", "—")
            try:
                m2.metric("Avg Precision", f"{average_precision_score(y_true, y_pred):.3f}")
            except Exception:
                m2.metric("Avg Precision", "—")
            try:
                m3.metric("Brier (lower=better)", f"{brier_score_loss(y_true, y_pred):.4f}")
            except Exception:
                m3.metric("Brier (lower=better)", "—")

        bins = st.slider("Calibration bins", min_value=5, max_value=20, value=10)
        df_tmp = df_scored.copy()
        df_tmp["bucket"] = pd.qcut(df_tmp["propensity"], q=bins, duplicates="drop")
        cal = (
            df_tmp.groupby("bucket", observed=True)
            .agg(avg_p=("propensity", "mean"), obs=(TARGET_COL, "mean"))
            .sort_values("avg_p")
            .reset_index(drop=True)
        )

        if not PLOTLY_OK:
            st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
            st.dataframe(cal, use_container_width=True)
        else:
            figc = go.Figure()
            figc.add_trace(go.Scatter(x=cal["avg_p"], y=cal["obs"], mode="lines+markers", name="Model"))
            figc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")))
            figc.update_layout(
                title="Reliability Curve (Predicted vs Observed)",
                xaxis_title="Mean predicted probability",
                yaxis_title="Observed conversion rate",
            )
            st.plotly_chart(figc, use_container_width=True)
