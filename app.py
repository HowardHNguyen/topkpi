# app.py
# Advanced Marketing KPI Performance with Data Science
# Streamlit app (root-level). Loads best_model_v2.pkl (joblib).
# Safe Plotly imports + guards; no sklearn shim code.

import os, time, sys, importlib.util, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ───────────────────────────────────────────────────────────────────────────────
# Try Plotly safely (app should not crash if Plotly is missing)
# ───────────────────────────────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    px, go, PLOTLY_OK = None, None, False

# safe loader: try joblib, fall back to pickle
try:
    import joblib
    HAVE_JOBLIB = True
except Exception:
    joblib = None
    HAVE_JOBLIB = False
import pickle

def load_any_pickle(path: str):
    if HAVE_JOBLIB:
        return joblib.load(path)
    # fallback to standard pickle
    with open(path, "rb") as f:
        return pickle.load(f)
    
# === Schema requirements ===
REQUIRED_FEATURES = [
    "State", "Coverage", "Education", "EmploymentStatus", "Income",
    "Location Code", "Marital Status", "Monthly Premium Auto",
    "Months Since Last Claim", "Months Since Policy Inception",
    "Number of Open Complaints", "Number of Policies",
    "Renew Offer Type", "Sales Channel", "Total Claim Amount",
    "Vehicle Class", "Vehicle Size"
]
OPTIONAL_FEATURES = [
    # Helpful but not required for scoring
    "conversion",                      # for back-tested KPIs, Lift, Calibration
    "Customer Lifetime Value"          # for realized CLV; proxy used if missing
]

# ───────────────────────────────────────────────────────────────────────────────
# App config
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Marketing KPI & Propensity Intelligence", page_icon="📊", layout="wide")

with st.expander("About This Project", expanded=False):
    st.markdown("""
    ... (content below) ...
    """)

DEFAULT_COST_MAP = {'Web': 40, 'Call Center': 70, 'Branch': 90, 'Agent': 120}
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
    st.write({
        "plotly": _ver("plotly"),
        "streamlit": _ver("streamlit"),
        "numpy": _ver("numpy"),
        "pandas": _ver("pandas"),
        "scikit_learn": _ver("sklearn"),
        "lightgbm": _ver("lightgbm"),
        "xgboost": _ver("xgboost"),
        "joblib": _ver("joblib"),
    })

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

def proxy_clv(df: pd.DataFrame) -> pd.Series:
    """Proxy CLV if CLV column is missing."""
    monthly = df.get("Monthly Premium Auto", pd.Series(np.nan, index=df.index)).astype(float)
    months, margin = 12.0, 0.35
    prox = months * monthly * margin
    if prox.isna().all():
        prox = pd.Series(12.0 * 100.0 * 0.35, index=df.index)  # fallback $420
    return prox

def compute_kpis(df: pd.DataFrame,
                 cost_map: dict = None,
                 channel_col: str = "Sales Channel",
                 offer_col: str = "Renew Offer Type",
                 clv_col: str = CLV_COL) -> dict:
    d = df.copy()
    cost_map = cost_map or DEFAULT_COST_MAP

    if channel_col in d.columns:
        d['est_acquisition_cost'] = d[channel_col].map(cost_map).fillna(np.median(list(cost_map.values())))
    else:
        d['est_acquisition_cost'] = np.median(list(cost_map.values()))

    clv = d[clv_col].clip(lower=0) if clv_col in d.columns else proxy_clv(d)

    conv_rate = float(d[TARGET_COL].mean()) if TARGET_COL in d.columns else np.nan
    acquired = int(d[TARGET_COL].sum()) if TARGET_COL in d.columns else np.nan

    total_cost = d.loc[d.get(TARGET_COL, pd.Series(False)) == 1, 'est_acquisition_cost'].sum()
    clv_realized = (clv * d.get(TARGET_COL, 0)).sum()
    cpa = float(total_cost / max(acquired, 1)) if acquired and acquired > 0 else np.nan
    roi = float((clv_realized - total_cost) / total_cost) if total_cost > 0 else np.nan

    out = {"overall": {
        "conversion_rate": conv_rate, "acquired": acquired, "cpa": cpa,
        "clv_realized": float(clv_realized), "roi": roi
    }}

    if channel_col in d.columns:
        g = d.groupby(channel_col).apply(lambda g: pd.Series({
            "conversion_rate": g.get(TARGET_COL, pd.Series(0)).mean(),
            "acquired": int(g.get(TARGET_COL, pd.Series(0)).sum()),
            "cpa": g.loc[g.get(TARGET_COL, pd.Series(False)) == 1, "est_acquisition_cost"].sum()
                    / max(g.get(TARGET_COL, pd.Series(0)).sum(), 1),
            "clv_realized": ((g[clv_col] if clv_col in g.columns else proxy_clv(g)) * g.get(TARGET_COL, 0)).sum(),
        }))
        g["roi"] = (g["clv_realized"] - g["cpa"]*g["acquired"]) / (g["cpa"]*g["acquired"]).replace(0, np.nan)
        out["by_channel"] = g.reset_index()

    if offer_col in d.columns:
        g2 = d.groupby(offer_col).apply(lambda g: pd.Series({
            "conversion_rate": g.get(TARGET_COL, pd.Series(0)).mean(),
            "acquired": int(g.get(TARGET_COL, pd.Series(0)).sum()),
            "cpa": g.loc[g.get(TARGET_COL, pd.Series(False)) == 1, "est_acquisition_cost"].sum()
                    / max(g.get(TARGET_COL, pd.Series(0)).sum(), 1),
            "clv_realized": ((g[clv_col] if clv_col in g.columns else proxy_clv(g)) * g.get(TARGET_COL, 0)).sum(),
        }))
        g2["roi"] = (g2["clv_realized"] - g2["cpa"]*g2["acquired"]) / (g2["cpa"]*g2["acquired"]).replace(0, np.nan)
        out["by_offer"] = g2.reset_index()

    return out

def lift_table(y_true: pd.Series, y_pred: pd.Series, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.values, "p": y_pred.values})
    df["bucket"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    agg = df.groupby("bucket").agg(
        n=("y", "size"),
        positives=("y", "sum"),
        avg_p=("p", "mean")
    ).sort_values("avg_p", ascending=False).reset_index(drop=True)
    agg["rate"] = agg["positives"] / agg["n"].replace(0, np.nan)
    agg["cum_positives"] = agg["positives"].cumsum()
    agg["cum_n"] = agg["n"].cumsum()
    agg["cum_rate"] = agg["cum_positives"] / agg["cum_n"].replace(0, np.nan)
    base = df["y"].mean() if df["y"].notna().any() else np.nan
    agg["lift"] = agg["rate"] / base if base and base > 0 else np.nan
    agg["cum_lift"] = agg["cum_rate"] / base if base and base > 0 else np.nan
    return agg

def fmt_pct(x): return "-" if pd.isna(x) else f"{100*x:.2f}%"
def fmt_money(x): return "-" if pd.isna(x) else f"${x:,.0f}"

# Additional notes for Green vs Red for checking csv file upload
def render_schema_checklist(df: pd.DataFrame | None):
    with st.sidebar.expander("Schema checklist", expanded=True):
        if df is None:
            st.info("Upload a CSV to validate the schema. Use `sample_data.csv` as a template.")
            st.caption(
                "Required features drive scoring; optional fields enable realized CLV/actual KPI back-tests."
            )
            st.code(
                "Required:\n" + ", ".join(REQUIRED_FEATURES) + "\n\n"
                "Optional:\n" + ", ".join(OPTIONAL_FEATURES),
                language="text",
            )
            return

        cols = set(df.columns)
        missing = [c for c in REQUIRED_FEATURES if c not in cols]
        extras  = sorted(list(cols.difference(set(REQUIRED_FEATURES + OPTIONAL_FEATURES))))

        ok = (len(missing) == 0)
        if ok:
            st.success("All required columns are present. ✅")
        else:
            st.error(f"Missing {len(missing)} required column(s).")

        # Compact checklist
        st.write("**Required features**")
        for c in REQUIRED_FEATURES:
            st.markdown(f"- {'✅' if c in cols else '❌'} `{c}`")

        st.write("**Optional features**")
        for c in OPTIONAL_FEATURES:
            st.markdown(f"- {'✅' if c in cols else '➖'} `{c}`")

        if extras:
            st.write("**Extra columns (ignored by the model)**")
            st.caption(", ".join(extras))

        # Quick tip panel
        with st.popover("Tips"):
            st.markdown(
                """
- Column names must match **exactly** (case & spacing).
- Categorical values not seen in training are handled safely.
- If `conversion` is missing, KPIs are **estimated** from scores until actuals are available.
- If `Customer Lifetime Value` is missing, the app uses a **conservative proxy** so ROI still works.
                """
            )


# ───────────────────────────────────────────────────────────────────────────────
# Model loader (no shims) — only load the clean artifact
# ───────────────────────────────────────────────────────────────────────────────
MODEL_CANDIDATES = ["best_model.pkl"]  # ensure this file is in the repo root

@st.cache_data(show_spinner=False)
def load_model(path: str):
    return load_any_pickle(path)

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
    cc  = st.number_input("Call Center", value=int(DEFAULT_COST_MAP["Call Center"]), min_value=0, step=5)
    br  = st.number_input("Branch", value=int(DEFAULT_COST_MAP["Branch"]), min_value=0, step=5)
    ag  = st.number_input("Agent", value=int(DEFAULT_COST_MAP["Agent"]), min_value=0, step=5)
user_cost_map = {'Web': web, 'Call Center': cc, 'Branch': br, 'Agent': ag}

section = st.sidebar.radio("Navigate",
                           ["📥 Data", "📊 KPIs", "🤖 Propensity", "📈 Lift & Gain", "🧪 Calibration"],
                           index=1)

if pipe is not None:
    st.sidebar.success(model_info)
else:
    st.sidebar.error(f"No model loaded. Ensure {MODEL_CANDIDATES[0]} is in the repo root.")

# ───────────────────────────────────────────────────────────────────────────────
# Data input
# ───────────────────────────────────────────────────────────────────────────────
st.title("Marketing KPI & Propensity Intelligence by Howard Nguyen")
st.write("**Conversion • CLV • CPA • ROI • Propensity** — actionable, executive-ready analytics with calibrated predictions.")

uploaded = st.file_uploader("Upload CSV (same schema as training)", type=["csv"], accept_multiple_files=False)

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists("sample_data.csv"):
    df = pd.read_csv("sample_data.csv")
    st.info("Loaded sample_data.csv from repo root. Upload your CSV to replace it.")
else:
    st.warning("Upload a CSV to begin (or include sample_data.csv in the repo root).")

if df is not None:
    st.write("**Preview**")
    st.dataframe(df.head(25), use_container_width=True)

has_target = df is not None and (TARGET_COL in df.columns)

# After df is set (or left as None)
render_schema_checklist(df)

# ───────────────────────────────────────────────────────────────────────────────
# KPIs
# ───────────────────────────────────────────────────────────────────────────────
if section == "📊 KPIs":
    st.subheader("KPI Overview")
    with st.expander("How to read this section", expanded=True):
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
    else:
        if has_target:
            kpis = compute_kpis(df, cost_map=user_cost_map)
            overall = kpis["overall"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Conversion Rate", fmt_pct(overall["conversion_rate"]))
            c2.metric("Acquired", overall["acquired"])
            c3.metric("CPA", fmt_money(overall["cpa"]))
            c4.metric("CLV Realized", fmt_money(overall["clv_realized"]))
            c5.metric("ROI", fmt_pct(overall["roi"]))
        else:
            st.info("No ground-truth conversion column found. Switch to **Propensity** to score first; you’ll see **Estimated KPIs** there.")

        if has_target and "by_channel" in kpis and isinstance(kpis["by_channel"], pd.DataFrame) and len(kpis["by_channel"]):
            by_ch = kpis["by_channel"]
            st.markdown("#### By Sales Channel")
            st.dataframe(by_ch, use_container_width=True)
            if not PLOTLY_OK:
                st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
                st.caption("**Conversion Rate by Channel** — Higher bars = more efficient funnel or better lead quality.")
            else:
                fig = px.bar(by_ch, x="Sales Channel", y="conversion_rate",
                             title="Conversion Rate by Channel",
                             text=by_ch["conversion_rate"].map(lambda v: f"{100*v:.1f}%"))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    """
                **Interpretation:**  
                - Use this to **shift spend**: scale channels with higher conversion and acceptable CPA.  
                - Combine with **ROI by Offer** to confirm profitability, not just efficiency.
                    """
                )

        if has_target and "by_offer" in kpis and isinstance(kpis["by_offer"], pd.DataFrame) and len(kpis["by_offer"]):
            by_offer = kpis["by_offer"]
            st.markdown("#### By Offer Type")
            st.dataframe(by_offer, use_container_width=True)
            if not PLOTLY_OK:
                st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
                st.caption("**ROI by Offer Type** — Bars show profitability after acquisition cost.")
            else:
                fig2 = px.bar(by_offer, x="Renew Offer Type", y="roi",
                              title="ROI by Offer Type",
                              text=by_offer["roi"].map(lambda v: f"{100*v:.1f}%"))
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown(
                    """
                **Interpretation:**  
                - **Pause** offers with near-zero ROI (spend returns ~$0).  
                - **Scale** offers with strong ROI **and** sufficient volume.  
                - If CPA increases (sidebar), check whether ROI still clears your hurdle.
                    """
                )

# ───────────────────────────────────────────────────────────────────────────────
# Propensity
# ───────────────────────────────────────────────────────────────────────────────
elif section == "🤖 Propensity":
    st.subheader("Propensity Scoring")
    with st.expander("How to read this section", expanded=True):
        st.markdown(
                """
        **Goal:** Rank customers by likelihood to convert and decide *how deep* to target.

        - **Top prospects** – highest probability first (use for outreach lists or routing).
        - **Propensity Distribution** – wider right tail = more “easy wins.”
        - **Threshold slider** – pick the **operating point** (e.g., top 20–30%).  
        The app shows **Estimated Converts** at that threshold.

        **Playbook**
        1. Start with a threshold that captures the **top deciles** (e.g., 0.6–0.7).
        2. Check **KPI estimates** (if ground truth missing) and **CPA** sensitivity.
        3. Expand or contract the target based on **capacity** and **ROI**.
                """
            )   
    if df is None:
        st.warning("Upload data to score.")
    elif pipe is None:
        st.warning(f"Model not loaded. Put {MODEL_CANDIDATES[0]} in repo root.")
    else:
        expected_cols = _expected_columns_from_pipeline(pipe)
        df_for_model = ensure_columns_for_pipeline(df, expected_cols)

        with st.spinner("Scoring…"):
            try:
                proba = pipe.predict_proba(df_for_model)[:, 1]
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                st.stop()

        df_scored = df.copy()
        df_scored["propensity"] = proba

        st.write("**Top prospects** (highest probability first):")
        st.dataframe(df_scored.sort_values("propensity", ascending=False).head(25), use_container_width=True)
        st.caption("**Propensity Distribution** — Right-skewed = model finds many high-probability customers.")

        colA, colB = st.columns([2, 1])
        with colA:
            if not PLOTLY_OK:
                st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
            else:
                figp = px.histogram(df_scored, x="propensity", nbins=30, title="Propensity Distribution")
                st.plotly_chart(figp, use_container_width=True)
        with colB:
            thr = st.slider("Decision threshold for estimated conversion", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            est_converted = int((df_scored["propensity"] >= thr).sum())
            st.metric("Estimated Converts @ threshold", est_converted)
            st.caption("**Threshold** — Sets how deep you go. Lower threshold = more volume, lower precision.")
            st.markdown(
                """
            **Note:** These are **estimated KPIs** using model scores (no ground truth).  
            Use them to **size campaigns** before you collect actual conversion outcomes.
                """
            )

        if TARGET_COL not in df_scored.columns:
            dtmp = df_scored.copy()
            dtmp[TARGET_COL] = (dtmp["propensity"] >= thr).astype(int)
            kpis_est = compute_kpis(dtmp, cost_map=user_cost_map)
            overall = kpis_est["overall"]
            st.markdown("#### Estimated KPIs (based on propensity ≥ threshold)")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Conversion Rate", fmt_pct(overall["conversion_rate"]))
            c2.metric("Acquired", overall["acquired"])
            c3.metric("CPA", fmt_money(overall["cpa"]))
            c4.metric("CLV Realized", fmt_money(overall["clv_realized"]))
            c5.metric("ROI", fmt_pct(overall["roi"]))
        st.markdown(
            """
        **Interpretation:**  
        - A **fat right tail** means you can target shallow (top deciles) and still hit goals.  
        - If the curve is flat, consider re-training with fresh features or better labels.
            """
        )
        csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored CSV", data=csv_bytes, file_name="scored_output.csv", mime="text/csv")

# ───────────────────────────────────────────────────────────────────────────────
# Lift & Gain
# ───────────────────────────────────────────────────────────────────────────────
elif section == "📈 Lift & Gain":
    st.subheader("Lift & Cumulative Gain")
    with st.expander("How to read this section", expanded=True):
        st.markdown(
            """
        **Goal:** Prove the model’s business value and decide how many deciles to target.  

        - **Lift** – conversion rate in a decile vs overall average (e.g., 6× in top decile).  
        - **Cumulative Lift** – lift as you include more deciles from the top down.

        **Rules of thumb**
        - Top decile lift **> 3×** is strong; **> 5×** is excellent.  
        - If lift collapses by decile 3–4, target fewer deciles (higher precision).
                """
        )

    if df is None:
        st.warning("Upload data to compute lift.")
    elif pipe is None:
        st.warning("Model not loaded.")
    elif TARGET_COL not in df.columns:
        st.info("Lift requires ground-truth conversion column.")
    else:
        expected_cols = _expected_columns_from_pipeline(pipe)
        df_for_model = ensure_columns_for_pipeline(df, expected_cols)
        proba = pipe.predict_proba(df_for_model)[:, 1]
        df_tmp = df.copy()
        df_tmp["propensity"] = proba

        lift = lift_table(df_tmp[TARGET_COL], df_tmp["propensity"], bins=10)
        st.dataframe(lift, use_container_width=True)
        st.caption("**Lift by Decile** — How much better each decile is vs targeting everyone.")

        if not PLOTLY_OK:
            st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")

        else:
            fig_lift = px.line(lift, y="lift", title="Lift by Decile")
            fig_lift.update_layout(xaxis_title="Decile (high→low propensity)", yaxis_title="Lift")
            st.caption("**Cumulative Lift** — What happens as you include more of the ranked list.")
            st.plotly_chart(fig_lift, use_container_width=True)

            fig_cum = px.line(lift, y="cum_lift", title="Cumulative Lift")
            fig_cum.update_layout(xaxis_title="Decile (cumulative)", yaxis_title="Cumulative Lift")
            st.plotly_chart(fig_cum, use_container_width=True)
            st.markdown(
                """
            **Interpretation:**  
            - A steep drop after decile 1–2 suggests a **tight focus** (smaller, higher-quality audience).  
            - A gradual slope suggests you can **scale** to more deciles without sharp efficiency loss.
                """
            )

# ───────────────────────────────────────────────────────────────────────────────
# Calibration
# ───────────────────────────────────────────────────────────────────────────────
elif section == "🧪 Calibration":
    st.subheader("Calibration Check (Binning)")
    with st.expander("How to read this section", expanded=True):
        st.markdown(
                """
        **Goal:** Verify that predicted probabilities match real-world outcomes.  

        - Blue line close to dashed **diagonal** = **well-calibrated**.  
        - **Overconfident** (above diagonal at low x): model predicts too high for low scores.  
        - **Underconfident** (below diagonal): model predicts too low.

        **Why it matters**
        - Good calibration makes **ROI forecasts** (score × CLV) trustworthy.  
        - If miscalibrated, re-calibrate (isotonic / Platt) or re-train with recent data.
                """
        )

    if df is None:
        st.warning("Upload data to check calibration.")
    elif pipe is None:
        st.warning("Model not loaded.")
    elif TARGET_COL not in df.columns:
        st.info("Calibration requires ground-truth conversion column.")
        st.caption("**Predicted vs Observed** — Dots close to the diagonal are reliable probabilities.")

    else:
        expected_cols = _expected_columns_from_pipeline(pipe)
        df_for_model = ensure_columns_for_pipeline(df, expected_cols)
        proba = pipe.predict_proba(df_for_model)[:, 1]
        df_tmp = df.copy()
        df_tmp["propensity"] = proba

        bins = st.slider("Calibration bins", min_value=5, max_value=20, value=10)
        df_tmp["bucket"] = pd.qcut(df_tmp["propensity"], q=bins, duplicates="drop")
        cal = df_tmp.groupby("bucket").agg(
            avg_p=("propensity", "mean"),
            obs=(TARGET_COL, "mean")
        ).sort_values("avg_p")

        if not PLOTLY_OK:
            st.warning("Plotly isn’t installed in this build. Add `plotly==5.24.1` to requirements.txt and redeploy.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cal["avg_p"], y=cal["obs"], mode="lines+markers", name="Model"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfectly calibrated", line=dict(dash="dash")))
            fig.update_layout(title="Calibration: Predicted vs Observed",
                              xaxis_title="Predicted probability", yaxis_title="Observed frequency")
            st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("© Howard Nguyen, PhD • KPI dashboard and calibrated propensity model. If CLV is missing, a proxy is used for ROI.")

### 💡 About This Project
This app and its accompanying article — **Advanced Marketing KPI Performance with Data Science** — were created to bridge the gap between traditional marketing metrics and modern AI intelligence.

In most marketing teams, KPIs such as **Conversion Rate (CR)**, **Customer Lifetime Value (CLV)**, **Cost per Acquisition (CPA)**, and **Return on Investment (ROI)** are analyzed separately.  
This project integrates them within a **predictive data-science framework** that reveals how they interact, enabling data-driven decision-making instead of static reporting.

---

### 🎯 **Purpose**
To empower marketing leaders, analysts, and growth teams to **move from descriptive to prescriptive insights**, using predictive models that forecast conversions, simulate ROI, and recommend optimal channel strategies.

---

### ⚙️ **How It Was Built**
- **Dataset:** Realistic customer and policy data (demographics, claims, sales channels, offers).
- **Modeling Stack:** `scikit-learn`, `LightGBM`, `XGBoost`, and a calibrated ensemble pipeline trained for **propensity modeling**.
- **Metrics:** CR, CLV, CPA, and ROI dynamically computed for every uploaded dataset.
- **Validation:** Stratified CV, grouped CV by customer, and permutation AUC to detect overfitting.
- **Visualization:** Interactive **Plotly** dashboards with “How to Read This Section” notes for clarity.
- **Deployment:** Streamlit + GitHub + Python 3.12 environment (`scikit-learn 1.6.1`, `lightgbm 4.5.0`, `joblib 1.4.2`).

---

### 🧠 **Key Methods**
1. **Data Engineering** – cleaning, encoding, and feature balancing (SMOTE & feature selection).  
2. **Predictive Modeling** – comparing RF, XGB, LGBM, CNN, and Stacking Gen AI for best calibrated AUC.  
3. **KPI Simulation Engine** – live formulas for CR, CLV, CPA, and ROI tied to sidebar sliders.  
4. **Explainable Visuals** – contextual notes for each KPI, Lift, Gain, and Calibration plot.  
5. **Schema Validation** – automatic checklist ensures any uploaded CSV matches model features.

---

### 🚀 **Why It Matters**
This project demonstrates how **data science transforms marketing KPIs** into actionable intelligence:  
- **Marketers** can instantly visualize ROI by channel and offer.  
- **Executives** can test “what-if” CPA or budget scenarios live.  
- **Analysts** can upload new data and replicate results seamlessly.

> **Analytics → Insight → Action.**  
> The article educates; the app operationalizes — together they turn metrics into measurable growth.

