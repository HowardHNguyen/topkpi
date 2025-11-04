# 📊 Advanced Marketing KPI Performance with Data Science

**Live App:** [Streamlit Dashboard](https://topkpi.streamlit.app/)
**Article:** *Advanced Marketing KPI Performance with Data Science*  

---

### 💡 Overview
This project unites marketing metrics — **Conversion Rate (CR)**, **Customer Lifetime Value (CLV)**, **Cost per Acquisition (CPA)**, and **Return on Investment (ROI)** — within a predictive machine-learning framework.

It transforms static KPI dashboards into **interactive, AI-driven insights** that allow marketing teams to forecast outcomes, optimize spend, and simulate ROI across offers, channels, and audiences.

---

### ⚙️ Features
- **Dynamic KPI Simulation** – live CR, CLV, CPA, and ROI updates with adjustable CPAs.  
- **Propensity Modeling** – calibrated LightGBM + ensemble models predict conversion likelihood.  
- **Lift & Gain Curves** – quantify model effectiveness by decile.  
- **Calibration Plot** – verify probability reliability for ROI forecasting.  
- **Schema Checklist** – ensures any uploaded dataset matches expected features.  
- **“How to Read This Section” Notes** – help non-technical managers interpret each chart.

---

### 🧠 Tech Stack
| Layer | Tools & Libraries |
|-------|-------------------|
| Language | Python 3.12 |
| ML | scikit-learn 1.6.1 · LightGBM 4.5.0 · XGBoost 2.1.1 |
| Visualization | Plotly Express · Streamlit |
| Deployment | Streamlit Cloud · GitHub Actions |

---

### 🔍 Methodology
1. **Data Preparation** – encode, impute, balance, and drop leak features.  
2. **Model Comparison** – RF, XGB, LGBM, CNN, and Stacking Gen AI evaluated via AUC/AP/Brier.  
3. **Calibration** – isotonic regression for probability reliability.  
4. **KPI Engine** – CR, CLV, CPA, ROI computed dynamically from predictions.  
5. **Dashboard UX** – Plotly visuals with embedded guidance for every chart.  

---

### 🚀 Why It Matters
Traditional dashboards show *what happened*.  
This app shows *what could happen next* — and how to improve it.  

It’s designed for marketing executives, analysts, and data scientists seeking **explainable AI** tools that bridge analytics and business outcomes.

> **Predict. Explain. Optimize. Repeat.**

---

### 🧾 License
MIT © 2025 Howard Nguyen, PhD (MaxAIS)
