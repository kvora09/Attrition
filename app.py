"""
╔══════════════════════════════════════════════════════════════╗
║   Microland — Employee Attrition Prediction Dashboard        ║
║   Streamlit App                                              ║
╠══════════════════════════════════════════════════════════════╣
║   Run  :  streamlit run app.py                               ║
║   Needs:  artifacts/  folder (run train.py first)            ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

# ── Core ──────────────────────────────────────────────────────────────────────
import os, json, pickle
import numpy as np
import pandas as pd

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Microland – Attrition AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]          { background-color:#0f1117; }
[data-testid="stSidebar"]                   { background-color:#12141f; }
h1,h2,h3,h4                                 { color:#e2e8f0 !important; }
p, li, label                                { color:#cbd5e1; }
.stMetric label                             { color:#94a3b8 !important; }
.stMetric [data-testid="stMetricValue"]     { color:#f1f5f9 !important; }
.stDataFrame th                             { background-color:#1e2235 !important;
                                              color:#a5b4fc !important; }
.stTabs [data-baseweb="tab"]                { color:#94a3b8; }
.stTabs [aria-selected="true"]             { color:#a5b4fc;
                                              border-bottom:2px solid #a5b4fc; }
.risk-critical { background:#7f1d1d; border:1px solid #ef4444;
                 border-radius:6px; padding:4px 12px;
                 color:#fca5a5; font-weight:700; display:inline-block; }
.risk-high     { background:#7c2d12; border:1px solid #f97316;
                 border-radius:6px; padding:4px 12px;
                 color:#fdba74; font-weight:700; display:inline-block; }
.risk-medium   { background:#713f12; border:1px solid #eab308;
                 border-radius:6px; padding:4px 12px;
                 color:#fde047; font-weight:700; display:inline-block; }
.risk-low      { background:#14532d; border:1px solid #22c55e;
                 border-radius:6px; padding:4px 12px;
                 color:#86efac; font-weight:700; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ── Plot theme ─────────────────────────────────────────────────────────────────
BG, FG, PANEL = "#0f1117", "white", "#1a1d2e"
PALETTE       = ["#7c6af7", "#f59e0b", "#10b981", "#ef4444", "#38bdf8"]
BAND_ORDER    = ["P1A","P1B","P1C","P2A","P2B","P2C","P3A","P3B","P4","P5","P6"]

def ax_style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#44446688")

def risk_badge(cat: str) -> tuple:
    """Return (html_badge, hex_color) for a risk category."""
    cfg = {
        "Critical": ('<span class="risk-critical">🔴 Critical</span>', "#ef4444"),
        "High":     ('<span class="risk-high">🟠 High</span>',         "#f97316"),
        "Medium":   ('<span class="risk-medium">🟡 Medium</span>',     "#eab308"),
        "Low":      ('<span class="risk-low">🟢 Low</span>',           "#22c55e"),
    }
    return cfg.get(str(cat), (cat, "#888"))


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACTS = "artifacts"

@st.cache_resource
def load_artifacts():
    try:
        model    = pickle.load(open(f"{ARTIFACTS}/best_model.pkl",  "rb"))
        scaler   = pickle.load(open(f"{ARTIFACTS}/scaler.pkl",      "rb"))
        encoders = pickle.load(open(f"{ARTIFACTS}/encoders.pkl",    "rb"))
        meta     = json.load(open(f"{ARTIFACTS}/model_meta.json",   "r"))
        return model, scaler, encoders, meta
    except FileNotFoundError:
        st.error("⚠️  Artifact files not found.  Run `python train.py` first.")
        st.stop()

@st.cache_data
def load_data():
    scored = pd.read_csv(f"{ARTIFACTS}/employees_scored.csv")
    raw    = pd.read_csv(f"{ARTIFACTS}/Microland_employees.csv")
    return scored, raw

model, scaler, encoders, meta = load_artifacts()
df_scored, df_raw             = load_data()

FEATURE_NAMES = meta["feature_names"]
BEST_NAME     = meta["best_model_name"]
NEEDS_SCALE   = meta.get("needs_scale", False)
THRESH        = meta.get("decision_threshold", 0.40)
CATEGORICAL   = meta.get("categorical_cols", [
    "Gender","MaritalStatus","Education","LevelBand",
    "Department","Location","ShiftType","OverTime","BusinessTravel",
])
RISK_THRESH   = meta.get("risk_thresholds",
                          {"Critical":0.45,"High":0.32,"Medium":0.18})


# ── KPIs ──────────────────────────────────────────────────────────────────────
total      = len(df_scored)
exits      = (df_scored["Attrition"] == "Yes").sum()
critical_n = (df_scored["RiskCategory"] == "Critical").sum()
high_plus  = (df_scored["RiskCategory"].isin(["High","Critical"])).sum()
attr_rate  = exits / total


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏢 Microland HR")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 EDA & Analytics",
        "🤖 Model Comparison",
        "🎯 Live Prediction",
        "👥 Employee Lookup",
    ])
    st.markdown("---")
    st.markdown(f"**Best Model:** `{BEST_NAME}`")
    st.markdown(f"**Test AUC:** `{meta['test_auc']:.4f}`")
    st.markdown(f"**CV AUC:** `{meta['cv_auc']:.4f}`")
    st.markdown(f"**F1 Score:** `{meta['test_f1']:.4f}`")
    st.markdown(f"**Recall:** `{meta.get('test_recall','–')}`")
    st.markdown(f"**Calibrated:** `{meta.get('is_calibrated', False)}`")
    st.markdown("---")
    st.markdown(f"👥 **{total:,}** employees")
    st.markdown(f"📉 Attrition: **{attr_rate:.1%}**")
    st.caption(f"Trained: {meta.get('trained_on','–')[:10]}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("🏢 Microland – Attrition Prediction Dashboard")
    st.markdown("ML-powered employee attrition risk scoring · 35+ features · 5 models compared")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Employees",   f"{total:,}")
    c2.metric("Historical Exits",  f"{exits:,}", delta=f"{attr_rate:.1%} rate")
    c3.metric("🔴 Critical Risk",  f"{critical_n:,}",
              delta=f"{critical_n/total:.1%} of workforce", delta_color="inverse")
    c4.metric("High + Critical",   f"{high_plus:,}",
              delta=f"{high_plus/total:.1%}", delta_color="inverse")
    c5.metric("Model AUC",         f"{meta['test_auc']:.4f}")

    st.markdown("---")

    # Avg risk by dept & risk breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Avg Risk by Department")
        dp = df_scored.groupby("Department")["AttritionProb"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax_style(ax, fig)
        colors = plt.cm.RdYlGn_r(dp.values / dp.values.max())
        bars   = ax.barh(dp.index, dp.values, color=colors, alpha=0.9)
        for b, v in zip(bars, dp.values):
            ax.text(v+0.003, b.get_y()+b.get_height()/2,
                    f"{v:.1%}", va="center", color=FG, fontsize=8)
        ax.set_xlabel("Avg Attrition Probability", color=FG)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Risk Category Breakdown")
        rc   = df_scored["RiskCategory"].value_counts().reindex(
               ["Low","Medium","High","Critical"]).fillna(0)
        cmap = {"Low":"#22c55e","Medium":"#eab308","High":"#f97316","Critical":"#ef4444"}
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor(BG)
        ax.pie(rc.values, labels=rc.index,
               colors=[cmap[l] for l in rc.index],
               autopct="%1.1f%%", textprops={"color":FG}, startangle=90,
               wedgeprops={"edgecolor":"#0f1117","linewidth":2})
        ax.set_facecolor(BG)
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Risk by Level Band")
        lv = (df_scored.groupby("LevelBand")["AttritionProb"].mean()
              .reindex([b for b in BAND_ORDER if b in df_scored["LevelBand"].unique()]))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax_style(ax, fig)
        ax.bar(lv.index, lv.values,
               color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(lv))), alpha=0.9)
        ax.set_ylabel("Avg Risk Score", color=FG)
        ax.tick_params(axis="x", rotation=30, colors=FG)
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Risk by Location")
        lc = df_scored.groupby("Location")["AttritionProb"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax_style(ax, fig)
        ax.bar(lc.index, lc.values,
               color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(lc))), alpha=0.9)
        ax.tick_params(axis="x", rotation=30, colors=FG)
        ax.set_ylabel("Avg Risk Score", color=FG)
        st.pyplot(fig); plt.close()

    # Persona callout
    st.markdown("---")
    st.subheader("🎯 Persona Validation")
    p1, p2, p3 = st.columns(3)

    ideal = df_scored[
        (df_scored["Gender"]           == "Female")  &
        (df_scored["MaritalStatus"]    == "Married") &
        (df_scored.get("Location", pd.Series()).eq("WFH") if "Location" in df_scored.columns else True) &
        (df_scored.get("PromotionsLast3Yrs", pd.Series(0)) >= 1) &
        (df_scored["OverTime"]         == "No")      &
        (df_scored["JobSatisfaction"]  >= 3)
    ] if "PromotionsLast3Yrs" in df_scored.columns else df_scored.head(0)

    risky = df_scored[
        (df_scored["MaritalStatus"]   == "Single")   &
        (df_scored["Department"]      == "Sales")     &
        (df_scored["MarketRateRatio"] < 0.85)         &
        (df_scored["OverTime"]        == "Yes")
    ] if "MarketRateRatio" in df_scored.columns else df_scored.head(0)

    p1.metric("💚 Married WFH Woman, Promoted",
              f"{ideal['AttritionProb'].mean():.1%}" if len(ideal) else "–")
    p2.metric("🔴 Single Sales, Underpaid, OT",
              f"{risky['AttritionProb'].mean():.1%}" if len(risky) else "–")
    p3.metric("📊 Company Average",
              f"{df_scored['AttritionProb'].mean():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 EDA & Analytics":
    st.title("🔍 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["📈 Feature Analysis", "🗺️ Heatmaps", "📊 Summary Stats"])

    with tab1:
        dim = st.selectbox("Analyze attrition rate by:", [
            "Gender","Department","LevelBand","Location","ShiftType",
            "MaritalStatus","OverTime","BusinessTravel",
            "JobSatisfaction","ManagerRating","WorkLifeBalance",
            "AwardsReceived","PromotionsLast3Yrs",
        ])

        col1, col2 = st.columns(2)
        with col1:
            grp    = df_raw.groupby(dim)["AttritionBinary"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax_style(ax, fig)
            colors = plt.cm.RdYlGn_r(grp.values / grp.values.max())
            bars   = ax.bar(grp.index.astype(str), grp.values, color=colors, alpha=0.9)
            for b, v in zip(bars, grp.values):
                ax.text(b.get_x()+b.get_width()/2, v+0.005,
                        f"{v:.1%}", ha="center", color=FG, fontsize=8)
            ax.set_ylabel("Attrition Rate", color=FG)
            ax.set_title(f"Attrition Rate by {dim}", color=FG)
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax_style(ax, fig)
            for lbl, g in df_raw.groupby("Attrition"):
                ax.hist(g["MarketRateRatio"], bins=35, alpha=0.65, density=True,
                        label=lbl, color={"Yes":"#ef4444","No":"#10b981"}[lbl])
            ax.axvline(1.0, color="white", ls="--", lw=1.5, label="Market Parity")
            ax.set_xlabel("Market Rate Ratio", color=FG)
            ax.set_title("Pay vs Market by Attrition", color=FG)
            ax.legend(facecolor=PANEL, labelcolor=FG, fontsize=8)
            st.pyplot(fig); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("WFH Days vs Attrition")
            wfh_rate = df_raw.groupby("WFH_DaysPerWeek")["AttritionBinary"].mean()
            fig, ax  = plt.subplots(figsize=(6, 3))
            ax_style(ax, fig)
            ax.plot(wfh_rate.index, wfh_rate.values, "o-",
                    color="#10b981", lw=2.5, ms=8)
            ax.fill_between(wfh_rate.index, wfh_rate.values,
                            alpha=0.15, color="#10b981")
            for x, y in zip(wfh_rate.index, wfh_rate.values):
                ax.text(x, y+0.005, f"{y:.1%}", ha="center", color=FG, fontsize=8)
            ax.set_xlabel("WFH Days / Week", color=FG)
            ax.set_ylabel("Attrition Rate", color=FG)
            st.pyplot(fig); plt.close()

        with col4:
            st.subheader("Overtime × Gender Attrition")
            pivot_ot = df_raw.pivot_table(
                values="AttritionBinary", index="Gender",
                columns="OverTime", aggfunc="mean"
            ) * 100
            fig, ax = plt.subplots(figsize=(6, 3))
            ax_style(ax, fig)
            x = np.arange(len(pivot_ot))
            w = 0.35
            for i, col_name in enumerate(pivot_ot.columns):
                ax.bar(x + i*w - w/2, pivot_ot[col_name].values, w,
                       label=f"OT {col_name}", alpha=0.85,
                       color=["#10b981","#ef4444"][i])
            ax.set_xticks(x); ax.set_xticklabels(pivot_ot.index, color=FG)
            ax.set_ylabel("Attrition %", color=FG)
            ax.legend(facecolor=PANEL, labelcolor=FG, fontsize=8)
            st.pyplot(fig); plt.close()

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Gender × Department")
            pivot = df_raw.pivot_table(
                values="AttritionBinary", index="Department",
                columns="Gender", aggfunc="mean"
            ) * 100
            fig, ax = plt.subplots(figsize=(8, 5))
            ax_style(ax, fig)
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                        ax=ax, linewidths=0.5, annot_kws={"size":9})
            ax.set_title("Attrition % — Gender × Dept", color=FG)
            ax.tick_params(colors=FG)
            st.pyplot(fig); plt.close()

        with c2:
            st.subheader("Level Band × Shift Type")
            pivot2 = df_raw.pivot_table(
                values="AttritionBinary", index="LevelBand",
                columns="ShiftType", aggfunc="mean"
            ) * 100
            pivot2 = pivot2.reindex([b for b in BAND_ORDER if b in pivot2.index])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax_style(ax, fig)
            sns.heatmap(pivot2, annot=True, fmt=".1f", cmap="YlOrRd",
                        ax=ax, linewidths=0.5, annot_kws={"size":9})
            ax.set_title("Attrition % — LevelBand × Shift", color=FG)
            ax.tick_params(colors=FG)
            st.pyplot(fig); plt.close()

        # Satisfaction × Dept
        st.subheader("Job Satisfaction by Department (Attrition vs Retained)")
        sat_dept = df_raw.groupby(["Department","Attrition"])["JobSatisfaction"].mean().unstack()
        fig, ax  = plt.subplots(figsize=(12, 4))
        ax_style(ax, fig)
        x, w = np.arange(len(sat_dept)), 0.4
        ax.bar(x-w/2, sat_dept.get("No",  [0]*len(sat_dept)), w,
               color="#10b981", label="Retained", alpha=0.9)
        ax.bar(x+w/2, sat_dept.get("Yes", [0]*len(sat_dept)), w,
               color="#ef4444", label="Left",     alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(sat_dept.index, rotation=20, color=FG, fontsize=8)
        ax.set_ylabel("Avg Job Satisfaction", color=FG)
        ax.legend(facecolor=PANEL, labelcolor=FG)
        ax.set_ylim(0, 5)
        st.pyplot(fig); plt.close()

    with tab3:
        num_cols = [
            "Age","Salary_Annual","MarketRateRatio","YearsAtCompany",
            "YearsSinceLastPromo","Leaves_90d","AwardsReceived",
            "ManagerRating","WorkLifeBalance","AvgSatisfaction",
            "WFH_DaysPerWeek","PromotionsLast3Yrs","HikePercentLast",
        ]
        existing = [c for c in num_cols if c in df_raw.columns]
        st.subheader("Summary Statistics")
        st.dataframe(df_raw[existing].describe().round(3), use_container_width=True)

        st.subheader("Correlation with Predicted Attrition Probability")
        corr_cols = [c for c in existing if c in df_scored.columns]
        corrs = (df_scored[corr_cols + ["AttritionProb"]]
                 .corr()["AttritionProb"]
                 .drop("AttritionProb")
                 .sort_values())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax_style(ax, fig)
        colors_c = ["#22c55e" if v < 0 else "#ef4444" for v in corrs.values]
        ax.barh(corrs.index, corrs.values, color=colors_c, alpha=0.9)
        ax.axvline(0, color="white", lw=0.8)
        ax.set_title("Feature Correlation with Predicted Attrition Prob", color=FG)
        ax.set_xlabel("Pearson Correlation", color=FG)
        st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")

    all_models = meta.get("all_models", {})
    if all_models:
        rows = [
            {
                "Model":     name,
                "CV AUC":   m["cv_auc"],
                "Test AUC": m["test_auc"],
                "F1":        m["test_f1"],
                "Recall":    m.get("test_recall", "–"),
                "Accuracy": m["test_acc"],
                "🏆":        "✅" if name == BEST_NAME else "",
            }
            for name, m in all_models.items()
        ]
        df_comp = pd.DataFrame(rows).sort_values("Test AUC", ascending=False)
        st.dataframe(df_comp.set_index("Model"), use_container_width=True)

        # Bar chart
        names_sorted = df_comp["Model"].tolist()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax_style(ax, fig)
        x, w = np.arange(len(names_sorted)), 0.20
        ax.bar(x-1.5*w, [all_models[n]["cv_auc"]              for n in names_sorted],
               w, color="#7c6af7", label="CV AUC",   alpha=0.9)
        ax.bar(x-0.5*w, [all_models[n]["test_auc"]            for n in names_sorted],
               w, color="#10b981", label="Test AUC", alpha=0.9)
        ax.bar(x+0.5*w, [all_models[n]["test_f1"]             for n in names_sorted],
               w, color="#f59e0b", label="F1",       alpha=0.9)
        ax.bar(x+1.5*w, [all_models[n].get("test_recall", 0)  for n in names_sorted],
               w, color="#ef4444", label="Recall",   alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ","\n") for n in names_sorted], color=FG, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title("Model Performance Comparison", color=FG)
        ax.legend(facecolor=PANEL, labelcolor=FG)
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        **Why `{BEST_NAME}` was selected:**
        - Highest CV AUC across 5 stratified folds (most generalizable)
        - Best balance of precision and recall on held-out test set
        - `class_weight="balanced"` handles 20% minority class natively

        **Decision threshold:** `{THRESH}` — lowered from 0.5 to improve recall on leavers\n        **Probability calibration:** `{meta.get("is_calibrated", False)}` — helps avoid inflated 0.90+ outputs
        """)

    # Probability distribution
    st.subheader("Predicted Probability Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax_style(ax, fig)
    for lbl, g in df_scored.groupby("Attrition"):
        ax.hist(g["AttritionProb"], bins=50, alpha=0.65, density=True,
                label=lbl, color={"Yes":"#ef4444","No":"#10b981"}[lbl])
    for thresh, color, label in [
        (RISK_THRESH["Medium"],  "#eab308", "Medium"),
        (RISK_THRESH["High"],    "#f97316", "High"),
        (RISK_THRESH["Critical"],"#ef4444", "Critical"),
    ]:
        ax.axvline(thresh, color=color, ls="--", lw=1.2,
                   label=f"{label} threshold ({thresh})")
    ax.set_xlabel("Predicted Attrition Probability", color=FG)
    ax.set_ylabel("Density", color=FG)
    ax.set_title("Model Output Distribution — Actual Attrition Overlay", color=FG)
    ax.legend(facecolor=PANEL, labelcolor=FG, fontsize=8)
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Live Prediction":
    st.title("🎯 Live Attrition Risk Prediction")
    st.markdown("Fill in an employee profile to get an instant AI-powered risk score.")

    with st.form("pred_form"):
        st.subheader("👤 Demographics")
        c1, c2, c3 = st.columns(3)
        age     = c1.slider("Age", 22, 60, 32)
        gender  = c2.selectbox("Gender", ["Male","Female","Other"])
        marital = c3.selectbox("Marital Status", ["Single","Married","Divorced"])
        edu     = c1.selectbox("Education", ["Bachelor","Master","PhD","Diploma","Other"])

        st.subheader("🏢 Org & Role")
        c4, c5, c6 = st.columns(3)
        band     = c4.selectbox("Level Band", BAND_ORDER)
        dept     = c5.selectbox("Department",
                                ["Engineering","Sales","Finance","HR","Marketing",
                                 "Operations","Legal","Product","QA","Infra"])
        location = c6.selectbox("Location",
                                ["Bengaluru","Mumbai","Hyderabad","Pune",
                                 "Chennai","Delhi","WFH"])
        shift    = c4.selectbox("Shift Type", ["Day","Night","Rotational","Flexible"])
        overtime = c5.selectbox("OverTime", ["No","Yes"])
        travel   = c6.selectbox("Business Travel", ["None","Rare","Frequent"])
        wfh_days = c4.slider("WFH Days / Week", 0, 5, 0)

        st.subheader("📅 Tenure & Career")
        c7, c8, c9 = st.columns(3)
        yrs_co    = c7.slider("Years at Company", 0, 35, 3)
        yrs_role  = c8.slider("Years in Current Role", 0, 20, 2)
        yrs_promo = c9.slider("Years Since Last Promo", 0, 12, 2)
        promos_3y = c7.slider("Promotions Last 3 Years", 0, 3, 0)
        num_cos   = c8.slider("Companies Worked", 0, 9, 2)
        dist_home = c9.slider("Distance from Home (km)", 1, 60, 15)

        st.subheader("💰 Compensation")
        c10, c11, c12 = st.columns(3)
        mkt_ratio = c10.slider("Market Rate Ratio (1.0 = parity)", 0.60, 1.40, 1.00, 0.01)
        hike_pct  = c11.slider("Last Increment %", 0, 30, 10)
        bonus_pct = c12.slider("Bonus %", 0, 40, 10)

        st.subheader("😊 Satisfaction & Performance  (1=Low · 4=High)")
        c13, c14, c15, c16 = st.columns(4)
        job_sat  = c13.selectbox("Job Satisfaction",  [1,2,3,4], index=2)
        env_sat  = c14.selectbox("Env Satisfaction",  [1,2,3,4], index=2)
        rel_sat  = c15.selectbox("Relationship Sat",  [1,2,3,4], index=2)
        wlb      = c16.selectbox("Work-Life Balance",  [1,2,3,4], index=2)
        mgr_rat  = c13.selectbox("Manager Rating",    [1,2,3,4,5], index=3)
        perf_rat = c14.selectbox("Performance Rating",[1,2,3,4], index=2)
        awards   = c15.slider("Awards Received", 0, 5, 0)
        recogn   = c16.slider("Recognitions", 0, 8, 1)
        training = c13.slider("Training Sessions (year)", 0, 6, 2)

        st.subheader("🏖️ Leave History")
        c17, c18, c19, c20 = st.columns(4)
        l30  = c17.slider("Leaves (30d)",  0, 10, 1)
        l90  = c18.slider("Leaves (90d)",  0, 15, 3)
        l180 = c19.slider("Leaves (180d)", 0, 25, 6)
        l365 = c20.slider("Leaves (365d)", 0, 40, 12)
        sick = c17.slider("Sick Leaves (annual)", 0, 12, 2)

        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        avg_sat  = (job_sat + env_sat + rel_sat + wlb) / 4.0
        band_num = {"P1A":1,"P1B":2,"P1C":3,"P2A":4,"P2B":5,"P2C":6,
                    "P3A":7,"P3B":8,"P4":9,"P5":10,"P6":11}[band]
        total_ex = max(age - 22, 0)

        rec = {
            "Age": age, "Gender": gender, "MaritalStatus": marital,
            "Education": edu, "LevelBand": band, "LevelBandNum": band_num,
            "Department": dept, "Location": location, "ShiftType": shift,
            "YearsAtCompany": yrs_co, "YearsInCurrentRole": yrs_role,
            "YearsWithCurrManager": min(yrs_role, yrs_co),
            "TotalWorkingYears": total_ex, "YearsSinceLastPromo": yrs_promo,
            "PromotionsLast3Yrs": promos_3y,
            "Salary_Annual": 1_000_000, "MarketRateRatio": mkt_ratio,
            "HikePercentLast": hike_pct, "BonusPct": bonus_pct,
            "ManagerRating": mgr_rat, "TeamSize": 10, "DistanceFromHome": dist_home,
            "JobSatisfaction": job_sat, "EnvSatisfaction": env_sat,
            "RelationshipSat": rel_sat, "WorkLifeBalance": wlb,
            "AvgSatisfaction": round(avg_sat, 2),
            "PerformanceRating": perf_rat, "AwardsReceived": awards,
            "RecognitionsReceived": recogn, "TrainingLastYear": training,
            "Leaves_30d": l30, "Leaves_90d": l90, "Leaves_180d": l180,
            "Leaves_365d": l365, "SickLeave_Annual": sick,
            "OverTime": overtime, "BusinessTravel": travel,
            "NumCompaniesWorked": num_cos, "WFH_DaysPerWeek": wfh_days,
            # Engineered features
            "IsUnderpaid":         int(mkt_ratio < 0.90),
            "IsSeverelyUnderpaid": int(mkt_ratio < 0.80),
            "PayGapPct":           round((1 - mkt_ratio) * 100, 1),
            "StagnationIndex":     max(yrs_promo * 2 - promos_3y * 3, 0),
            "RecognitionGap":      max(perf_rat - min(awards, 4), 0),
            "LeaveSpike30":        int(l30 >= 3),
            "LeaveSpike90":        int(l90 >= 8),
            "LeaveVelocity":       round(l365 / max(yrs_co, 1), 2),
            "SatComposite":        round(job_sat*0.35+wlb*0.30+env_sat*0.20+rel_sat*0.15, 2),
            "SatBelow2":           int(avg_sat < 2.0),
            "OT_LowSat":          int(overtime == "Yes" and job_sat <= 2),
            "WFH_SatInteraction":  wfh_days * job_sat,
            "TravelDistance":      dist_home if travel == "Frequent" else 0,
            "IsNewJoiner":         int(yrs_co <= 1),
            "IsVeteran":           int(yrs_co >= 10),
            "IsJobHopper":         int(num_cos >= 4),
            "CompanyLoyaltyScore": round(yrs_co / max(total_ex, 1), 2),
            "IsDisruptiveShift":   int(shift in ["Night","Rotational"]),
            "IsFullWFH":           int(wfh_days == 5),
            "IsMarried":           int(marital == "Married"),
            "CompositeRiskFlags":  (int(mkt_ratio < 0.90) + int(yrs_promo > 4) +
                                    int(overtime == "Yes") + int(l30 >= 3) +
                                    int(shift in ["Night","Rotational"])),
        }

        input_df = pd.DataFrame([rec])
        for col in CATEGORICAL:
            if col in input_df.columns and col in encoders:
                le  = encoders[col]
                val = str(input_df[col].iloc[0])
                input_df[col] = le.transform([val])[0] if val in le.classes_ else -1

        for f in FEATURE_NAMES:
            if f not in input_df.columns:
                input_df[f] = 0
        X_in = input_df[FEATURE_NAMES].values
        if NEEDS_SCALE:
            X_in = scaler.transform(X_in)

        prob = model.predict_proba(X_in)[0, 1]
        cat  = ("Critical" if prob >= RISK_THRESH["Critical"] else
                "High"     if prob >= RISK_THRESH["High"]     else
                "Medium"   if prob >= RISK_THRESH["Medium"]   else "Low")

        badge_html, badge_color = risk_badge(cat)

        st.markdown("---")
        st.subheader("Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Attrition Probability", f"{prob:.1%}")
        r2.markdown("**Risk Category**")
        r2.markdown(badge_html, unsafe_allow_html=True)
        r3.metric("Model Confidence", f"{max(prob, 1-prob):.1%}")

        # Gauge
        fig, ax = plt.subplots(figsize=(8, 2.5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
        for lo, hi, c in [(0, 0.18,"#22c55e"),(0.18,0.32,"#eab308"),
                           (0.32,0.45,"#f97316"),(0.45,1.0,"#ef4444")]:
            ax.barh(0, hi-lo, left=lo, height=0.5, color=c, alpha=0.75)
        ax.barh(0, 0.006, left=prob-0.003, height=0.65, color="white")
        ax.text(prob, 0.42, f"{prob:.1%}", ha="center",
                color=FG, fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xlabel("Attrition Probability", color=FG)
        ax.tick_params(colors=FG)
        for x, lbl in [(0.09,"Low"),(0.25,"Medium"),(0.385,"High"),(0.725,"Critical")]:
            ax.text(x, -0.38, lbl, ha="center", color=FG, fontsize=8)
        [sp.set_edgecolor("#444466") for sp in ax.spines.values()]
        st.pyplot(fig); plt.close()

        # Risk & retention factors
        st.subheader("Risk & Retention Factors Detected")
        cf, cr = st.columns(2)

        factors = []
        if mkt_ratio < 0.85: factors.append(f"🔴 **Underpaid vs market** ({mkt_ratio:.0%})")
        if overtime == "Yes": factors.append("🔴 **Overtime** — strongest burnout signal")
        if yrs_promo >= 4:    factors.append(f"🟠 **Career stagnant** — no promo in {yrs_promo}yr")
        if job_sat <= 2:      factors.append(f"🟠 **Low job satisfaction** ({job_sat}/4)")
        if wlb <= 2:          factors.append(f"🟠 **Poor work-life balance** ({wlb}/4)")
        if travel == "Frequent": factors.append("🟡 **Frequent business travel**")
        if shift in ["Night","Rotational"]: factors.append(f"🟡 **{shift} shift**")
        if yrs_co <= 1:       factors.append("🟡 **New joiner** — first-year risk")
        if num_cos >= 4:      factors.append(f"🟡 **Job hopper** — {num_cos} companies")

        retention = []
        if marital == "Married":  retention.append("💚 Married — stability factor")
        if wfh_days >= 3:         retention.append(f"💚 WFH {wfh_days}d/wk — flexibility")
        if promos_3y >= 1:        retention.append(f"💚 {promos_3y} promotion(s) last 3yr")
        if awards >= 2:           retention.append(f"💚 {awards} awards — recognition")
        if mkt_ratio >= 1.05:     retention.append(f"💚 Above-market pay ({mkt_ratio:.0%})")
        if job_sat >= 4:          retention.append("💚 High job satisfaction")
        if yrs_co >= 10:          retention.append(f"💚 Veteran — {yrs_co}yr tenure")

        with cf:
            st.markdown("**⚠️ Risk Factors:**")
            if factors:
                for f in factors: st.markdown(f)
            else:
                st.markdown("✅ No major risk factors identified")

        with cr:
            st.markdown("**🛡️ Retention Factors:**")
            if retention:
                for r in retention: st.markdown(r)
            else:
                st.markdown("⚠️ No strong retention factors found")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EMPLOYEE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👥 Employee Lookup":
    st.title("👥 Employee Risk Lookup")

    col1, col2, col3 = st.columns(3)
    dept_f = col1.selectbox("Department", ["All"] + sorted(df_scored["Department"].unique()))
    risk_f = col2.multiselect("Risk Category", ["Critical","High","Medium","Low"],
                               default=["Critical","High"])
    band_f = col3.multiselect("Level Band",
                               [b for b in BAND_ORDER if b in df_scored["LevelBand"].unique()])

    filt = df_scored.copy()
    if dept_f != "All":      filt = filt[filt["Department"] == dept_f]
    if risk_f:               filt = filt[filt["RiskCategory"].isin(risk_f)]
    if band_f:               filt = filt[filt["LevelBand"].isin(band_f)]
    filt = filt.sort_values("AttritionProb", ascending=False)

    st.markdown(f"Showing **{len(filt):,}** employees")

    display_cols = ["EmployeeID","Gender","Department","LevelBand","Location",
                    "ShiftType","OverTime","MarketRateRatio","JobSatisfaction",
                    "YearsSinceLastPromo","AttritionProb","RiskCategory","Attrition"]
    display_cols = [c for c in display_cols if c in filt.columns]

    st.dataframe(
        filt[display_cols].head(200)
            .style
            .background_gradient(subset=["AttritionProb"], cmap="RdYlGn_r")
            .format({"AttritionProb": "{:.1%}", "MarketRateRatio": "{:.2f}"}),
        use_container_width=True,
        height=450,
    )

    # Employee deep dive
    st.markdown("---")
    st.subheader("Employee Deep Dive")
    if len(filt) > 0:
        sel_id = st.selectbox("Select Employee ID", filt["EmployeeID"].tolist()[:50])
        emp    = filt[filt["EmployeeID"] == sel_id].iloc[0]
        badge_html, _ = risk_badge(emp["RiskCategory"])

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Attrition Prob", f"{emp['AttritionProb']:.1%}")
        d2.markdown("**Risk:**"); d2.markdown(badge_html, unsafe_allow_html=True)
        d3.metric("Market Rate",
                  f"{emp.get('MarketRateRatio', '–'):.2f}"
                  if "MarketRateRatio" in emp.index else "–")
        d4.metric("Job Satisfaction",
                  f"{emp.get('JobSatisfaction','–')}/4")

        show_fields = [
            "Department","LevelBand","Location","ShiftType","OverTime",
            "YearsAtCompany","YearsSinceLastPromo","PromotionsLast3Yrs",
            "WorkLifeBalance","ManagerRating","AwardsReceived","Attrition",
        ]
        detail = {f: emp[f] for f in show_fields if f in emp.index}
        st.table(pd.Series(detail).rename("Value"))
    else:
        st.info("No employees match the selected filters.")
