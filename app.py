# app.py (patched for Streamlit number_input types)

import os, json, time, numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go, joblib
st.set_page_config(page_title="MaxAIS • KPI + Propensity", page_icon="📈", layout="wide")

DEFAULT_COST_MAP = {'Web': 40, 'Call Center': 70, 'Branch': 90, 'Agent': 120}
TARGET_COL = "conversion"
CLV_COL = "Customer Lifetime Value"

def _expected_columns_from_pipeline(pipe):
    try:
        pre = pipe.named_steps.get("pre") or pipe.named_steps.get("pre3")
        cols = []
        for _, _, cols_sel in pre.transformers_:
            if cols_sel == "drop": continue
            if isinstance(cols_sel, list): cols.extend(cols_sel)
            elif isinstance(cols_sel, (tuple, np.ndarray, pd.Index)): cols.extend(list(cols_sel))
        return list(dict.fromkeys(cols))
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_pickle(path): return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_csv(path): return pd.read_csv(path)

def ensure_columns_for_pipeline(df, expected):
    if expected is None: return df
    out = df.copy()
    for c in expected:
        if c not in out.columns: out[c] = np.nan
    return out[expected]

def proxy_clv(df):
    monthly = df.get("Monthly Premium Auto", pd.Series(np.nan, index=df.index)).astype(float)
    months, margin = 12.0, 0.35
    prox = months * monthly * margin
    if prox.isna().all(): prox = pd.Series(12.0 * 100.0 * 0.35, index=df.index)
    return prox

def compute_kpis(df, cost_map=None, channel_col="Sales Channel", offer_col="Renew Offer Type", clv_col=CLV_COL):
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
    out = {"overall": {"conversion_rate": conv_rate, "acquired": acquired, "cpa": cpa, "clv_realized": float(clv_realized), "roi": roi}}
    if channel_col in d.columns:
        g = d.groupby(channel_col).apply(lambda g: pd.Series({
            "conversion_rate": g.get(TARGET_COL, pd.Series(0)).mean(),
            "acquired": int(g.get(TARGET_COL, pd.Series(0)).sum()),
            "cpa": g.loc[g.get(TARGET_COL, pd.Series(False)) == 1, "est_acquisition_cost"].sum() / max(g.get(TARGET_COL, pd.Series(0)).sum(), 1),
            "clv_realized": ((g[CLV_COL] if CLV_COL in g.columns else proxy_clv(g)) * g.get(TARGET_COL, 0)).sum(),
        }))
        g["roi"] = (g["clv_realized"] - g["cpa"]*g["acquired"]) / (g["cpa"]*g["acquired"]).replace(0, np.nan)
        out["by_channel"] = g.reset_index()
    if offer_col in d.columns:
        g2 = d.groupby(offer_col).apply(lambda g: pd.Series({
            "conversion_rate": g.get(TARGET_COL, pd.Series(0)).mean(),
            "acquired": int(g.get(TARGET_COL, pd.Series(0)).sum()),
            "cpa": g.loc[g.get(TARGET_COL, pd.Series(False)) == 1, "est_acquisition_cost"].sum() / max(g.get(TARGET_COL, pd.Series(0)).sum(), 1),
            "clv_realized": ((g[CLV_COL] if CLV_COL in g.columns else proxy_clv(g)) * g.get(TARGET_COL, 0)).sum(),
        }))
        g2["roi"] = (g2["clv_realized"] - g2["cpa"]*g2["acquired"]) / (g2["cpa"]*g2["acquired"]).replace(0, np.nan)
        out["by_offer"] = g2.reset_index()
    return out

def lift_table(y_true, y_pred, bins=10):
    df = pd.DataFrame({"y": y_true.values, "p": y_pred.values})
    df["bucket"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    agg = df.groupby("bucket").agg(n=("y","size"), positives=("y","sum"), avg_p=("p","mean")).sort_values("avg_p", ascending=False).reset_index(drop=True)
    agg["rate"] = agg["positives"] / agg["n"].replace(0, np.nan)
    agg["cum_positives"] = agg["positives"].cumsum(); agg["cum_n"] = agg["n"].cumsum()
    agg["cum_rate"] = agg["cum_positives"] / agg["cum_n"].replace(0, np.nan)
    base = df["y"].mean() if df["y"].notna().any() else np.nan
    agg["lift"] = agg["rate"] / base if base and base > 0 else np.nan
    agg["cum_lift"] = agg["cum_rate"] / base if base and base > 0 else np.nan
    return agg

def fmt_pct(x): return "-" if pd.isna(x) else f"{100*x:.2f}%"
def fmt_money(x): return "-" if pd.isna(x) else f"${x:,.0f}"

# Sidebar
st.sidebar.title("MaxAIS • KPI + Propensity")
st.sidebar.caption("Advanced Marketing KPI Performance — Conversion • CLV • CPA • ROI • Propensity")
with st.sidebar.expander("Cost per Acquisition (override)", expanded=False):
    # 🔧 FIX: keep ALL args int type (value/min_value/step)
    web = st.number_input("Web", value=int(DEFAULT_COST_MAP["Web"]), min_value=0, step=5)
    cc  = st.number_input("Call Center", value=int(DEFAULT_COST_MAP["Call Center"]), min_value=0, step=5)
    br  = st.number_input("Branch", value=int(DEFAULT_COST_MAP["Branch"]), min_value=0, step=5)
    ag  = st.number_input("Agent", value=int(DEFAULT_COST_MAP["Agent"]), min_value=0, step=5)
user_cost_map = {'Web': web, 'Call Center': cc, 'Branch': br, 'Agent': ag}

section = st.sidebar.radio("Navigate", ["📥 Data", "📊 KPIs", "🤖 Propensity", "📈 Lift & Gain", "🧪 Calibration"], index=1)

# Load model
MODEL_PATH = "best_model.pkl"
pipe = load_pickle(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
if pipe is None: st.sidebar.warning("best_model.pkl not found in repo root. Upload it to enable scoring.")

# Data input
st.title("Advanced Marketing KPI Performance with Data Science")
st.write("**Conversion • CLV • CPA • ROI • Propensity** — ready for executive review and action.")
uploaded = st.file_uploader("Upload CSV (same schema as training)", type=["csv"], accept_multiple_files=False)
df = pd.read_csv(uploaded) if uploaded is not None else (load_csv("sample_data.csv") if os.path.exists("sample_data.csv") else None)
if df is not None:
    st.write("**Preview**"); st.dataframe(df.head(25), use_container_width=True)
else:
    st.warning("Upload a CSV to begin (or include sample_data.csv in the repo root).")
has_target = df is not None and (TARGET_COL in df.columns)

# KPIs
if section == "📊 KPIs":
    st.subheader("KPI Overview")
    if df is None:
        st.warning("Upload data to compute KPIs.")
    else:
        if has_target:
            kpis = compute_kpis(df, cost_map=user_cost_map)
            overall = kpis["overall"]
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Conversion Rate", fmt_pct(overall["conversion_rate"]))
            c2.metric("Acquired", overall["acquired"])
            c3.metric("CPA", fmt_money(overall["cpa"]))
            c4.metric("CLV Realized", fmt_money(overall["clv_realized"]))
            c5.metric("ROI", fmt_pct(overall["roi"]))
        else:
            st.info("No ground-truth conversion column found. Switch to **Propensity** to score first, then **Estimated KPIs** will be shown there.")
        if has_target and "by_channel" in kpis:
            by_ch = kpis["by_channel"]; st.markdown("#### By Sales Channel"); st.dataframe(by_ch, use_container_width=True)
            fig = px.bar(by_ch, x="Sales Channel", y="conversion_rate", title="Conversion Rate by Channel",
                         text=by_ch["conversion_rate"].map(lambda v: f"{100*v:.1f}%")); st.plotly_chart(fig, use_container_width=True)
        if has_target and "by_offer" in kpis:
            by_offer = kpis["by_offer"]; st.markdown("#### By Offer Type"); st.dataframe(by_offer, use_container_width=True)
            fig2 = px.bar(by_offer, x="Renew Offer Type", y="roi", title="ROI by Offer Type",
                          text=by_offer["roi"].map(lambda v: f"{100*v:.1f}%")); st.plotly_chart(fig2, use_container_width=True)

# Propensity
elif section == "🤖 Propensity":
    st.subheader("Propensity Scoring")
    if df is None: st.warning("Upload data to score.")
    elif pipe is None: st.warning("Model not loaded. Put best_model.pkl in repo root.")
    else:
        expected_cols = _expected_columns_from_pipeline(pipe)
        df_for_model = ensure_columns_for_pipeline(df, expected_cols)
        with st.spinner("Scoring…"):
            proba = pipe.predict_proba(df_for_model)[:, 1]
        df_scored = df.copy(); df_scored["propensity"] = proba
        st.write("**Top prospects** (highest probability first):")
        st.dataframe(df_scored.sort_values("propensity", ascending=False).head(25), use_container_width=True)
        colA, colB = st.columns([2, 1])
        with colA:
            figp = px.histogram(df_scored, x="propensity", nbins=30, title="Propensity Distribution")
            st.plotly_chart(figp, use_container_width=True)
        with colB:
            thr = st.slider("Decision threshold for estimated conversion", 0.0, 1.0, 0.5, 0.01)
            est_converted = int((df_scored["propensity"] >= thr).sum()); st.metric("Estimated Converts @ threshold", est_converted)
        if TARGET_COL not in df_scored.columns:
            dtmp = df_scored.copy(); dtmp[TARGET_COL] = (dtmp["propensity"] >= thr).astype(int)
            kpis_est = compute_kpis(dtmp, cost_map=user_cost_map)
            overall = kpis_est["overall"]
            st.markdown("#### Estimated KPIs (based on propensity ≥ threshold)")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Conversion Rate", fmt_pct(overall["conversion_rate"]))
            c2.metric("Acquired", overall["acquired"])
            c3.metric("CPA", fmt_money(overall["cpa"]))
            c4.metric("CLV Realized", fmt_money(overall["clv_realized"]))
            c5.metric("ROI", fmt_pct(overall["roi"]))
        csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored CSV", data=csv_bytes, file_name="scored_output.csv", mime="text/csv")

# Lift & Gain
elif section == "📈 Lift & Gain":
    st.subheader("Lift & Cumulative Gain")
    if df is None: st.warning("Upload data to compute lift.")
    elif pipe is None: st.warning("Model not loaded.")
    elif TARGET_COL not in df.columns: st.info("Lift requires ground-truth conversion column.")
    else:
        expected_cols = _expected_columns_from_pipeline(pipe)
        df_for_model = ensure_columns_for_pipeline(df, expected_cols)
        proba = pipe.predict_proba(df_for_model)[:, 1]
        df_tmp = df.copy(); df_tmp["propensity"] = proba
        lift = lift_table(df_tmp[TARGET_COL], df_tmp["propensity"], bins=10)
        st.dataframe(lift, use_container_width=True)
        fig_lift = px.line(lift, y="lift", title="Lift by Decile"); fig_lift.update_layout(xaxis_title="Decile (high→low propensity)", yaxis_title="Lift")
        st.plotly_chart(fig_lift, use_container_width=True)
        fig_cum = px.line(lift, y="cum_lift", title="Cumulative Lift"); fig_cum.update_layout(xaxis_title="Decile (cumulative)", yaxis_title="Cumulative Lift")
        st.plotly_chart(fig_cum, use_container_width=True)

# Calibration
elif section == "🧪 Calibration":
    st.subheader("Calibration Check (Binning)")
    if df is None: st.warning("Upload data to check calibration.")
    elif pipe is None: st.warning("Model not loaded.")
    elif TARGET_COL not in df.columns: st.info("Calibration requires ground-truth conversion column.")
    else:
        expected_cols = _expected_columns_from_pipeline(pipe)
        df_for_model = ensure_columns_for_pipeline(df, expected_cols)
        proba = pipe.predict_proba(df_for_model)[:, 1]
        df_tmp = df.copy(); df_tmp["propensity"] = proba
        bins = st.slider("Calibration bins", 5, 20, 10)
        df_tmp["bucket"] = pd.qcut(df_tmp["propensity"], q=bins, duplicates="drop")
        cal = df_tmp.groupby("bucket").agg(avg_p=("propensity","mean"), obs=("conversion","mean")).sort_values("avg_p")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal["avg_p"], y=cal["obs"], mode="lines+markers", name="Model"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfectly calibrated", line=dict(dash="dash")))
        fig.update_layout(title="Calibration: Predicted vs Observed", xaxis_title="Predicted probability", yaxis_title="Observed frequency")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("© Howard Nguyen, PhD • KPI dashboard and calibrated propensity model. If CLV is missing, a proxy is used for ROI.")