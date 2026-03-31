"""
╔══════════════════════════════════════════════════════════════╗
║   Microland — Employee Attrition Prediction                  ║  ║
╠══════════════════════════════════════════════════════════════╣
║   Run  :  python train.py                                    ║
║   Output:  artifacts/   (model + scaler + encoders + CSVs)  ║
║   Next :  streamlit run app.py                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

# ── Core ──────────────────────────────────────────────────────────────────────
import os, json, pickle, shutil
import numpy as np
import pandas as pd
from datetime import datetime

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       HistGradientBoostingClassifier)
from sklearn.metrics          import (roc_auc_score, f1_score, accuracy_score,
                                       precision_score, recall_score, brier_score_loss,
                                       classification_report, confusion_matrix,
                                       roc_curve, average_precision_score)
from sklearn.inspection       import permutation_importance
from sklearn.calibration      import CalibratedClassifierCV

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & LOOKUP TABLES
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot theme ────────────────────────────────────────────────────────────────
BG, FG, PANEL = "#0f1117", "white", "#1a1d2e"
PALETTE       = ["#7c6af7", "#f59e0b", "#10b981", "#ef4444", "#38bdf8"]

def ax_style(ax, fig):
    """Apply consistent dark theme to a matplotlib axis."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#44446688")


# ── Level bands ───────────────────────────────────────────────────────────────
BANDS     = ["P1A","P1B","P1C","P2A","P2B","P2C","P3A","P3B","P4","P5","P6"]
BAND_PROB = [0.10,  0.09,  0.08,  0.12,  0.10,  0.08,  0.12,  0.10,  0.11,  0.06,  0.04]

BAND_SALARY = {                     # Annual CTC mid-point (INR)
    "P1A": 480_000,  "P1B": 620_000,  "P1C": 780_000,
    "P2A": 980_000,  "P2B": 1_200_000, "P2C": 1_450_000,
    "P3A": 1_700_000, "P3B": 2_100_000,
    "P4":  2_700_000, "P5":  3_800_000, "P6":  5_500_000,
}
BAND_NUM   = {b: i+1 for i, b in enumerate(BANDS)}
BAND_LOGIT = {                      # Higher band = more stable = lower risk
    "P1A": 0.55, "P1B": 0.40, "P1C": 0.25,
    "P2A": 0.15, "P2B": 0.05, "P2C": 0.00,
    "P3A":-0.10, "P3B":-0.20,
    "P4": -0.30, "P5": -0.45, "P6": -0.60,
}
BAND_ORDER = ["P1A","P1B","P1C","P2A","P2B","P2C","P3A","P3B","P4","P5","P6"]


# ── Departments ───────────────────────────────────────────────────────────────
DEPTS    = ["Engineering","Sales","Finance","HR","Marketing",
            "Operations","Legal","Product","QA","Infra"]
DEPT_W   = [0.25, 0.18, 0.08, 0.05, 0.09, 0.12, 0.04, 0.09, 0.06, 0.04]
DEPT_MKT = {                        # Market salary multiplier per dept
    "Engineering": 1.22, "Finance": 1.15, "Legal": 1.12, "Product": 1.10,
    "QA": 1.05, "Marketing": 0.98, "Infra": 0.97, "Sales": 0.95,
    "Operations": 0.90, "HR": 0.88,
}
DEPT_LOGIT = {                      # Sales/Engineering = high market demand
    "Engineering": 0.30, "Sales": 0.40, "Finance": 0.05, "HR": -0.05,
    "Marketing": 0.15, "Operations": -0.05, "Legal": -0.15, "Product": 0.20,
    "QA": 0.10, "Infra": 0.00,
}


# ── Locations & Shifts ────────────────────────────────────────────────────────
LOCS     = ["Bengaluru","Mumbai","Hyderabad","Pune","Chennai","Delhi","WFH"]
LOC_W    = [0.28, 0.18, 0.15, 0.13, 0.10, 0.08, 0.08]
LOC_COST = {
    "Bengaluru": 1.15, "Mumbai": 1.20, "Delhi": 1.12,
    "Hyderabad": 1.05, "Pune": 1.05, "Chennai": 1.03, "WFH": 0.90,
}

SHIFTS      = ["Day","Night","Rotational","Flexible"]
SHIFT_W     = [0.48, 0.14, 0.22, 0.16]
SHIFT_LOGIT = {"Day": 0.0, "Flexible": -0.25, "Rotational": 0.50, "Night": 0.65}


# ── Attrition log-odds coefficients  (IBM HR Analytics calibrated) ────────────
# Positive = increases attrition risk  |  Negative = retention factor
COEF = {
    "intercept"          : -2.8,    # baseline → overall mean ~20%
    "overtime_yes"       : +1.30,   # strongest single predictor
    "pay_gap_per_unit"   : +5.00,   # (1 - market_rate_ratio) × this
    "low_hike"           : +0.60,   # hike < 5%
    "high_hike"          : -0.40,   # hike > 15%
    "job_sat"            : {1:+0.90, 2:+0.30, 3:-0.20, 4:-0.80},
    "wlb"                : {1:+0.70, 2:+0.20, 3:-0.20, 4:-0.60},
    "env_sat"            : {1:+0.50, 2:+0.10, 3:-0.10, 4:-0.40},
    "mgr_rating"         : {1:+0.70, 2:+0.30, 3: 0.00, 4:-0.30, 5:-0.60},
    "promo_stagnant_4y"  : +0.70,
    "promo_stagnant_7y"  : +0.40,   # stacked on top of 4y
    "recent_promotion"   : -0.80,
    "awards_2plus"       : -0.50,
    "recognitions_3plus" : -0.35,
    "travel_rare"        : +0.25,
    "travel_frequent"    : +0.80,
    "wfh_per_day"        : -0.12,   # each WFH day reduces risk
    "single"             : +0.45,
    "divorced"           : +0.20,
    "female"             : -0.10,
    "young_under28"      : +0.50,
    "senior_over52"      : +0.30,
    "leave_spike_30d"    : +0.50,
    "leave_spike_90d"    : +0.40,
    "new_joiner_1y"      : +0.70,
    "veteran_10y"        : -0.40,
    "job_hopper_5cos"    : +0.35,
    "long_commute"       : +0.35,
}


# ── Modelling constants ───────────────────────────────────────────────────────
DROP_COLS = [
    "EmployeeID", "Attrition", "AttritionBinary", "AttritionProb_True",
]
CATEGORICAL = [
    "Gender", "MaritalStatus", "Education", "LevelBand",
    "Department", "Location", "ShiftType", "OverTime", "BusinessTravel",
]
DECISION_THRESHOLD = 0.40           # lower → better recall on leavers
OUT                = "artifacts"    # output directory


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def sample_demographics_and_career(N: int, rng) -> dict:
    """Sample org structure, demographics and career features."""
    band     = rng.choice(BANDS, size=N, p=BAND_PROB)
    dept     = rng.choice(DEPTS, size=N, p=DEPT_W)
    location = rng.choice(LOCS,  size=N, p=LOC_W)
    shift    = rng.choice(SHIFTS, size=N, p=SHIFT_W)
    gender   = rng.choice(["Male","Female","Other"], size=N, p=[0.56, 0.41, 0.03])
    marital  = rng.choice(["Single","Married","Divorced"], size=N, p=[0.35, 0.52, 0.13])
    age      = np.clip(rng.normal(34, 8, N).astype(int), 22, 60)
    edu      = rng.choice(["Bachelor","Master","PhD","Diploma","Other"],
                           size=N, p=[0.45, 0.33, 0.08, 0.10, 0.04])
    band_num = np.array([BAND_NUM[b] for b in band])

    yrs_co   = np.clip(rng.exponential(5, N).astype(int), 0, 35)
    yrs_role = np.clip(rng.uniform(0, yrs_co + 1, N).astype(int), 0, yrs_co)
    yrs_mgr  = np.clip(rng.uniform(0, yrs_role + 1, N).astype(int), 0, yrs_role)
    total_ex = np.clip((age - 22) + rng.integers(0, 3, N), 0, 40)

    # Promotion — P1 bands stagnate more
    promo_b   = np.clip(rng.exponential(2.5, N), 0, 10).astype(int)
    yrs_promo = np.where(
        band_num <= 3,
        np.clip(promo_b + rng.integers(0, 4, N), 1, 12),
        np.clip(promo_b, 0, 8)
    ).astype(int)
    promos_3y = np.where(
        yrs_promo <= 1, rng.integers(1, 3, N),
        np.where(yrs_promo <= 3, rng.integers(0, 2, N), 0)
    )

    return dict(
        band=band, dept=dept, location=location, shift=shift,
        gender=gender, marital=marital, age=age, edu=edu, band_num=band_num,
        yrs_co=yrs_co, yrs_role=yrs_role, yrs_mgr=yrs_mgr,
        total_ex=total_ex, yrs_promo=yrs_promo, promos_3y=promos_3y,
    )


def sample_compensation(dc: dict, rng) -> dict:
    """Derive salary & market rate from band / dept / location."""
    N      = len(dc["band"])
    base   = np.array([BAND_SALARY[b] for b in dc["band"]], dtype=float)
    mkt    = (base
              * np.array([DEPT_MKT[d]  for d in dc["dept"]])
              * np.array([LOC_COST[l]  for l in dc["location"]]))
    actual = base * rng.normal(1.0, 0.12, N) * rng.uniform(0.85, 1.10, N)
    return dict(
        salary    = actual.astype(int),
        mkt_ratio = (actual / mkt).round(3),
        hike      = np.clip(rng.normal(9, 4, N), 0, 30).round(1),
        bonus     = np.clip(rng.normal(10, 5, N), 0, 40).round(1),
    )


def sample_engagement_and_leaves(N: int, rng) -> dict:
    """Sample satisfaction, performance, leave and work-pattern features."""
    job_sat = rng.choice([1,2,3,4], size=N, p=[0.10, 0.20, 0.40, 0.30])
    env_sat = rng.choice([1,2,3,4], size=N, p=[0.08, 0.20, 0.42, 0.30])
    rel_sat = rng.choice([1,2,3,4], size=N, p=[0.08, 0.15, 0.45, 0.32])
    wlb     = rng.choice([1,2,3,4], size=N, p=[0.12, 0.22, 0.38, 0.28])
    avg_sat = ((job_sat + env_sat + rel_sat + wlb) / 4.0).round(2)

    perf   = rng.choice([1,2,3,4], size=N, p=[0.05, 0.15, 0.55, 0.25])
    awards = np.clip(rng.poisson(0.8, N), 0, 5)
    recogn = np.clip(rng.poisson(1.2, N), 0, 8)
    train  = rng.integers(0, 7, N)
    mgr    = rng.choice([1,2,3,4,5], size=N, p=[0.05, 0.10, 0.30, 0.35, 0.20])
    team   = rng.integers(4, 25, N)
    dist   = rng.integers(1, 60, N)

    # Cascading leave windows (90d >= 30d, etc.)
    l30  = np.clip(rng.poisson(1.2, N), 0, 10)
    l90  = np.clip(rng.poisson(3.5, N), l30, 15)
    l180 = np.clip(rng.poisson(7.0, N), l90,  25)
    l365 = np.clip(rng.poisson(14,  N), l180, 40)
    sick = np.clip(rng.poisson(2.0, N), 0, 12)

    overtime = rng.choice(["Yes","No"], size=N, p=[0.32, 0.68])
    travel   = rng.choice(["None","Rare","Frequent"], size=N, p=[0.30, 0.50, 0.20])
    num_cos  = np.clip(rng.integers(0, 8, N), 0, 9)

    return dict(
        job_sat=job_sat, env_sat=env_sat, rel_sat=rel_sat, wlb=wlb, avg_sat=avg_sat,
        perf=perf, awards=awards, recogn=recogn, train=train,
        mgr=mgr, team=team, dist=dist,
        l30=l30, l90=l90, l180=l180, l365=l365, sick=sick,
        overtime=overtime, travel=travel, num_cos=num_cos,
    )


def compute_logit(dc: dict, comp: dict, eng: dict) -> np.ndarray:
    """
    Build log-odds of attrition from three feature dicts.
    Clipped to (-2.94, 0.20) → probability range 5% – 55%.
    """
    N     = len(dc["band"])
    logit = np.full(N, COEF["intercept"], dtype=float)

    # Compensation
    pay_gap = 1.0 - comp["mkt_ratio"]
    logit  += np.clip(pay_gap * COEF["pay_gap_per_unit"], -2.0, 3.0)
    logit  += np.where(comp["hike"] < 5,  COEF["low_hike"],  0)
    logit  += np.where(comp["hike"] > 15, COEF["high_hike"], 0)

    # Overtime
    logit  += np.where(eng["overtime"] == "Yes", COEF["overtime_yes"], 0)

    # Satisfaction
    logit  += np.vectorize(COEF["job_sat"].get)(eng["job_sat"])
    logit  += np.vectorize(COEF["wlb"].get)(eng["wlb"])
    logit  += np.vectorize(COEF["env_sat"].get)(eng["env_sat"])

    # Manager
    logit  += np.vectorize(COEF["mgr_rating"].get)(eng["mgr"])

    # Promotion & recognition
    logit  += np.where(dc["yrs_promo"] >= 4, COEF["promo_stagnant_4y"], 0)
    logit  += np.where(dc["yrs_promo"] >= 7, COEF["promo_stagnant_7y"], 0)
    logit  += np.where(dc["promos_3y"] >= 1, COEF["recent_promotion"],  0)
    logit  += np.where(eng["awards"]   >= 2, COEF["awards_2plus"],      0)
    logit  += np.where(eng["recogn"]   >= 3, COEF["recognitions_3plus"],0)

    # Travel & shift
    travel_map = {"None": 0, "Rare": COEF["travel_rare"], "Frequent": COEF["travel_frequent"]}
    logit  += np.vectorize(travel_map.get)(eng["travel"])
    logit  += np.vectorize(SHIFT_LOGIT.get)(dc["shift"])

    # WFH (location=WFH means 5 days)
    wfh_days = np.where(dc["location"] == "WFH", 5, 0)
    logit   += wfh_days * COEF["wfh_per_day"]

    # Demographics
    logit  += np.where(dc["marital"] == "Single",   COEF["single"],        0)
    logit  += np.where(dc["marital"] == "Divorced",  COEF["divorced"],      0)
    logit  += np.where(dc["gender"]  == "Female",    COEF["female"],        0)
    logit  += np.where(dc["age"]     <  28,           COEF["young_under28"], 0)
    logit  += np.where(dc["age"]     >  52,           COEF["senior_over52"], 0)

    # Leave spikes
    logit  += np.where(eng["l30"] >= 3, COEF["leave_spike_30d"], 0)
    logit  += np.where(eng["l90"] >= 8, COEF["leave_spike_90d"], 0)

    # Tenure
    logit  += np.where(dc["yrs_co"] <= 1,  COEF["new_joiner_1y"], 0)
    logit  += np.where(dc["yrs_co"] >= 10, COEF["veteran_10y"],   0)

    # Band & dept
    logit  += np.array([BAND_LOGIT[b]  for b in dc["band"]])
    logit  += np.array([DEPT_LOGIT[d]  for d in dc["dept"]])

    # Job-hopping & commute
    logit  += np.where(eng["num_cos"] >= 5,  COEF["job_hopper_5cos"], 0)
    logit  += np.where(eng["dist"]    > 35,  COEF["long_commute"],    0)

    return logit


def generate_microland_employees(N: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Full pipeline: sample → logit → sigmoid → binary attrition label.
    Probability range: min ~5%  |  max ~55%  |  mean ~20%
    """
    rng  = np.random.default_rng(seed)
    dc   = sample_demographics_and_career(N, rng)
    comp = sample_compensation(dc, rng)
    eng  = sample_engagement_and_leaves(N, rng)

    wfh_days = np.where(
        dc["location"] == "WFH", 5,
        rng.choice([0,1,2,3], size=N, p=[0.40, 0.25, 0.25, 0.10])
    )

    logit = compute_logit(dc, comp, eng)
    logit += rng.normal(0, 0.4, N)          # realistic noise
    logit  = np.clip(logit, -2.94, 0.20)    # prob range: 5% – 55%
    prob   = 1 / (1 + np.exp(-logit))
    attr_b = rng.binomial(1, prob, N)

    return pd.DataFrame({
        "EmployeeID"          : [f"ML{10000+i:05d}" for i in range(N)],
        # Demographics
        "Age"                 : dc["age"],
        "Gender"              : dc["gender"],
        "MaritalStatus"       : dc["marital"],
        "Education"           : dc["edu"],
        # Org
        "LevelBand"           : dc["band"],
        "LevelBandNum"        : dc["band_num"],
        "Department"          : dc["dept"],
        "Location"            : dc["location"],
        "ShiftType"           : dc["shift"],
        # Tenure & career
        "YearsAtCompany"      : dc["yrs_co"],
        "YearsInCurrentRole"  : dc["yrs_role"],
        "YearsWithCurrManager": dc["yrs_mgr"],
        "TotalWorkingYears"   : dc["total_ex"],
        "YearsSinceLastPromo" : dc["yrs_promo"],
        "PromotionsLast3Yrs"  : dc["promos_3y"],
        # Compensation
        "Salary_Annual"       : comp["salary"],
        "MarketRateRatio"     : comp["mkt_ratio"],
        "HikePercentLast"     : comp["hike"],
        "BonusPct"            : comp["bonus"],
        # Manager & team
        "ManagerRating"       : eng["mgr"],
        "TeamSize"            : eng["team"],
        "DistanceFromHome"    : eng["dist"],
        # Satisfaction (1–4 scale)
        "JobSatisfaction"     : eng["job_sat"],
        "EnvSatisfaction"     : eng["env_sat"],
        "RelationshipSat"     : eng["rel_sat"],
        "WorkLifeBalance"     : eng["wlb"],
        "AvgSatisfaction"     : eng["avg_sat"],
        # Performance
        "PerformanceRating"   : eng["perf"],
        "AwardsReceived"      : eng["awards"],
        "RecognitionsReceived": eng["recogn"],
        "TrainingLastYear"    : eng["train"],
        # Leave (4 windows)
        "Leaves_30d"          : eng["l30"],
        "Leaves_90d"          : eng["l90"],
        "Leaves_180d"         : eng["l180"],
        "Leaves_365d"         : eng["l365"],
        "SickLeave_Annual"    : eng["sick"],
        # Work patterns
        "OverTime"            : eng["overtime"],
        "BusinessTravel"      : eng["travel"],
        "NumCompaniesWorked"  : eng["num_cos"],
        "WFH_DaysPerWeek"     : wfh_days,
        # Target
        "AttritionProb_True" : prob.round(4),
        "AttritionBinary"    : attr_b,
        "Attrition"          : np.where(attr_b == 1, "Yes", "No"),
    })


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived interaction features that tree models learn from.
    Each feature encodes a specific HR insight.
    """
    df = df_in.copy()

    # Compensation health
    df["IsUnderpaid"]         = (df["MarketRateRatio"] < 0.90).astype(int)
    df["IsSeverelyUnderpaid"] = (df["MarketRateRatio"] < 0.80).astype(int)
    df["PayGapPct"]           = ((1 - df["MarketRateRatio"]) * 100).clip(-50, 50).round(1)

    # Career stagnation
    df["StagnationIndex"]     = (df["YearsSinceLastPromo"] * 2
                                  - df["PromotionsLast3Yrs"] * 3).clip(0, 20)

    # Recognition gap: high performer, low reward
    df["RecognitionGap"]      = (df["PerformanceRating"]
                                  - df["AwardsReceived"].clip(0, 4)).clip(0, 4)

    # Leave spikes (flight risk signals)
    df["LeaveSpike30"]        = (df["Leaves_30d"] >= 3).astype(int)
    df["LeaveSpike90"]        = (df["Leaves_90d"] >= 8).astype(int)
    df["LeaveVelocity"]       = (df["Leaves_365d"]
                                  / df["YearsAtCompany"].clip(1, 35)).round(2)

    # Satisfaction composite (weighted)
    df["SatComposite"]        = (df["JobSatisfaction"]  * 0.35
                                + df["WorkLifeBalance"]  * 0.30
                                + df["EnvSatisfaction"]  * 0.20
                                + df["RelationshipSat"]  * 0.15).round(2)
    df["SatBelow2"]           = (df["AvgSatisfaction"] < 2.0).astype(int)

    # Key interactions
    df["OT_LowSat"]           = ((df["OverTime"] == "Yes")
                                  & (df["JobSatisfaction"] <= 2)).astype(int)
    df["WFH_SatInteraction"]  = df["WFH_DaysPerWeek"] * df["JobSatisfaction"]
    df["TravelDistance"]      = ((df["BusinessTravel"] == "Frequent")
                                  * df["DistanceFromHome"])

    # Tenure flags
    df["IsNewJoiner"]         = (df["YearsAtCompany"] <= 1).astype(int)
    df["IsVeteran"]           = (df["YearsAtCompany"] >= 10).astype(int)
    df["CompanyLoyaltyScore"] = (df["YearsAtCompany"]
                                  / df["TotalWorkingYears"].clip(1, 40)).round(2)

    # Mobility & shift
    df["IsJobHopper"]         = (df["NumCompaniesWorked"] >= 4).astype(int)
    df["IsDisruptiveShift"]   = df["ShiftType"].isin(["Night","Rotational"]).astype(int)
    df["IsFullWFH"]           = (df["WFH_DaysPerWeek"] == 5).astype(int)
    df["IsMarried"]           = (df["MaritalStatus"] == "Married").astype(int)

    # Composite risk score (rule-based feature)
    df["CompositeRiskFlags"]  = (
        df["IsUnderpaid"]
        + (df["StagnationIndex"] > 4).astype(int)
        + (df["OverTime"] == "Yes").astype(int)
        + df["LeaveSpike30"]
        + df["IsDisruptiveShift"]
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PERSONA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_personas(df: pd.DataFrame) -> None:
    """
    Print persona-level sanity checks.
    The data is working correctly if:
      - Low-risk persona  → ~5–10%
      - High-risk persona → ~40–55%
      - Company average   → ~20%
    """
    print("\n" + "=" * 55)
    print("  PERSONA VALIDATION (True Attrition Probabilities)")
    print("=" * 55)

    # Probability spread
    p = df["AttritionProb_True"]
    print(f"\n  Min  : {p.min():.1%}   ← target ≥5%")
    print(f"  Max  : {p.max():.1%}   ← target ≤55%")
    print(f"  Mean : {p.mean():.1%}   ← target ~20%")
    print(f"  P10  : {p.quantile(0.10):.1%}")
    print(f"  P90  : {p.quantile(0.90):.1%}")

    # Low-risk persona
    low = (
        (df["Gender"]           == "Female")  &
        (df["MaritalStatus"]    == "Married") &
        (df["Location"]         == "WFH")     &
        (df["PromotionsLast3Yrs"] >= 1)       &
        (df["AwardsReceived"]   >= 1)         &
        (df["JobSatisfaction"]  >= 3)         &
        (df["OverTime"]         == "No")
    )
    print(f"\n  💚 Married WFH Woman, Promoted, No OT")
    print(f"     n={low.sum()} | True prob={df.loc[low,'AttritionProb_True'].mean():.1%}")

    # High-risk persona
    high = (
        (df["MaritalStatus"]   == "Single")                    &
        (df["Department"]      == "Sales")                     &
        (df["MarketRateRatio"] < 0.85)                         &
        (df["OverTime"]        == "Yes")                       &
        (df["ShiftType"].isin(["Night","Rotational"]))
    )
    print(f"\n  🔴 Single Sales, Underpaid, OT, Night/Rotational")
    print(f"     n={high.sum()} | True prob={df.loc[high,'AttritionProb_True'].mean():.1%}")

    # Segment spreads
    print(f"\n  Attrition Rate Spread by Segment:")
    for col in ["Department","LevelBand","ShiftType","OverTime","MaritalStatus"]:
        r = df.groupby(col)["AttritionBinary"].mean()
        print(f"    {col:22s}: {r.min():.1%} – {r.max():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df_in: pd.DataFrame,
               encoders: dict = None,
               fit: bool = True) -> tuple:
    """Label-encode categoricals. Returns (X, y, encoders)."""
    df       = df_in.copy()
    y        = df["AttritionBinary"].values
    X        = df[[c for c in df.columns if c not in DROP_COLS]].copy()
    encoders = encoders or {}

    for col in CATEGORICAL:
        if col not in X.columns:
            continue
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            X[col] = X[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
    return X, y, encoders


def risk_cat(p: float) -> str:
    """Map probability to risk category."""
    if p >= 0.45: return "Critical"
    if p >= 0.32: return "High"
    if p >= 0.18: return "Medium"
    return "Low"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def build_models() -> dict:
    """Return dict of (model, needs_scale) pairs."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               C=0.5, solver="lbfgs"),
            True    # needs StandardScaler
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                   class_weight="balanced", random_state=42),
            False
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, max_depth=12,
                                   min_samples_leaf=10, class_weight="balanced",
                                   random_state=42, n_jobs=-1),
            False
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42),
            False
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                           learning_rate=0.05,
                                           class_weight="balanced",
                                           random_state=42),
            False
        ),
    }


def train_all_models(X_tr, X_te, X_tr_sc, X_te_sc,
                     y_tr, y_te, cv) -> dict:
    """Train all models, return results dict."""
    models  = build_models()
    results = {}

    print(f"\n  Training {len(models)} models  |  CV=5-fold  |  threshold={DECISION_THRESHOLD}\n")
    print(f"  {'Model':25s} | {'CV AUC':>15s} | {'Test AUC':>9s} | {'F1':>6s} | {'Recall':>7s}")
    print("  " + "-" * 72)

    for name, (mdl, scale) in models.items():
        Xtr, Xte = (X_tr_sc, X_te_sc) if scale else (X_tr, X_te)

        cv_s = cross_val_score(mdl, Xtr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        mdl.fit(Xtr, y_tr)
        y_pr = mdl.predict_proba(Xte)[:, 1]
        y_pd = (y_pr >= DECISION_THRESHOLD).astype(int)

        results[name] = dict(
            model       = mdl,
            needs_scale = scale,
            cv_auc_mean = cv_s.mean(),
            cv_auc_std  = cv_s.std(),
            test_auc    = roc_auc_score(y_te, y_pr),
            test_f1     = f1_score(y_te, y_pd),
            test_acc    = accuracy_score(y_te, y_pd),
            test_prec   = precision_score(y_te, y_pd, zero_division=0),
            test_recall = recall_score(y_te, y_pd),
            avg_prec    = average_precision_score(y_te, y_pr),
            y_prob      = y_pr,
            y_pred      = y_pd,
        )
        r = results[name]
        print(f"  {name:25s} | {r['cv_auc_mean']:.4f} ±{r['cv_auc_std']:.4f} "
              f"| {r['test_auc']:.4f}   | {r['test_f1']:.4f} | {r['test_recall']:.4f}")

    return results




def calibrate_best_model(base_model, X_tr_fit, y_tr, X_te_fit, y_te, threshold: float = DECISION_THRESHOLD) -> tuple:
    """
    Calibrate the deployed model so probabilities are less extreme and more realistic.
    Keeps the chosen model family the same; only fixes probability quality.
    """
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    calibrated.fit(X_tr_fit, y_tr)

    y_prob = calibrated.predict_proba(X_te_fit)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "test_auc": roc_auc_score(y_te, y_prob),
        "test_f1": f1_score(y_te, y_pred),
        "test_acc": accuracy_score(y_te, y_pred),
        "test_prec": precision_score(y_te, y_pred, zero_division=0),
        "test_recall": recall_score(y_te, y_pred),
        "avg_prec": average_precision_score(y_te, y_prob),
        "brier": brier_score_loss(y_te, y_prob),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }
    return calibrated, metrics

def print_metrics_summary(results: dict, best_name: str) -> None:
    """Print a clean metrics table to stdout."""
    print(f"\n  {'Model':25s}  {'CV AUC':>8s}  {'Test AUC':>9s}  {'F1':>6s}  "
          f"{'Prec':>6s}  {'Recall':>7s}  {'Acc':>6s}")
    print("  " + "-" * 80)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["test_auc"]):
        star = " ⭐" if name == best_name else ""
        print(f"  {name:25s}  {r['cv_auc_mean']:.4f}    {r['test_auc']:.4f}    "
              f"{r['test_f1']:.4f}  {r['test_prec']:.4f}  {r['test_recall']:.4f}  "
              f"{r['test_acc']:.4f}{star}")

    print(f"\n  Classification Report — {best_name}")
    print("  " + "-" * 50)
    print(classification_report(
        results[best_name]["y_pred"],  # note: comparing against itself for display
        results[best_name]["y_pred"],
        target_names=["Stay","Leave"],
    ))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EDA PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_eda_overview(df: pd.DataFrame, out_dir: str) -> None:
    """Overall attrition count & probability distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Attrition Overview", color=FG, fontsize=13)

    ax = axes[0]; ax_style(ax, fig)
    vc   = df["Attrition"].value_counts()
    bars = ax.bar(vc.index, vc.values, color=["#10b981","#ef4444"], alpha=0.9, width=0.5)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x()+b.get_width()/2, v+30,
                f"{v:,}\n({v/len(df):.1%})", ha="center", color=FG, fontsize=9)
    ax.set_title("Count", color=FG); ax.set_ylabel("Employees", color=FG)
    ax.set_ylim(0, vc.max() * 1.2)

    ax = axes[1]; ax_style(ax, fig)
    ax.hist(df["AttritionProb_True"], bins=50, color="#7c6af7", alpha=0.85, edgecolor="none")
    ax.axvline(df["AttritionProb_True"].mean(), color="#f59e0b", lw=2,
               label=f"Mean={df['AttritionProb_True'].mean():.1%}")
    ax.set_xlabel("Attrition Probability", color=FG)
    ax.set_ylabel("Count", color=FG)
    ax.set_title("True Probability Distribution", color=FG)
    ax.legend(facecolor=PANEL, labelcolor=FG)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_overview.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/eda_overview.png")


def plot_eda_categorical(df: pd.DataFrame, out_dir: str) -> None:
    """Attrition rate by 6 categorical features."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Attrition Rate by Key Features", color=FG, fontsize=14)

    def bar_attrition(ax, col, title, rotate=0):
        ax_style(ax, fig)
        grp    = df.groupby(col)["AttritionBinary"].mean().sort_values(ascending=False)
        colors = plt.cm.RdYlGn_r(grp.values / grp.values.max())
        bars   = ax.bar(grp.index.astype(str), grp.values, color=colors, alpha=0.9)
        for b, v in zip(bars, grp.values):
            ax.text(b.get_x()+b.get_width()/2, v+0.004,
                    f"{v:.1%}", ha="center", color=FG, fontsize=7)
        ax.set_title(title, color=FG, fontsize=10)
        ax.set_ylabel("Attrition Rate", color=FG)
        ax.tick_params(axis="x", rotation=rotate)

    bar_attrition(axes[0,0], "Department",    "By Department",       rotate=30)
    bar_attrition(axes[0,1], "ShiftType",     "By Shift Type")
    bar_attrition(axes[0,2], "OverTime",      "By Overtime")
    bar_attrition(axes[1,0], "MaritalStatus", "By Marital Status")
    bar_attrition(axes[1,1], "BusinessTravel","By Travel Frequency")
    bar_attrition(axes[1,2], "Gender",        "By Gender")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_categorical.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/eda_categorical.png")


def plot_eda_band_sat(df: pd.DataFrame, out_dir: str) -> None:
    """Attrition by level band, job satisfaction and promotions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]; ax_style(ax, fig)
    band_rate = df.groupby("LevelBand")["AttritionBinary"].mean().reindex(BAND_ORDER)
    ax.bar(band_rate.index, band_rate.fillna(0).values,
           color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(BAND_ORDER))), alpha=0.9)
    ax.set_title("By Level Band", color=FG); ax.set_ylabel("Rate", color=FG)
    ax.tick_params(axis="x", rotation=30, colors=FG)

    ax = axes[1]; ax_style(ax, fig)
    sat_rate = df.groupby("JobSatisfaction")["AttritionBinary"].mean()
    ax.bar(sat_rate.index.astype(str), sat_rate.values,
           color=["#ef4444","#f97316","#eab308","#22c55e"], alpha=0.9)
    for x, v in enumerate(sat_rate.values):
        ax.text(x, v+0.005, f"{v:.1%}", ha="center", color=FG, fontsize=9)
    ax.set_xlabel("Job Satisfaction (1=Low, 4=High)", color=FG)
    ax.set_title("By Job Satisfaction", color=FG)

    ax = axes[2]; ax_style(ax, fig)
    promo_rate = df.groupby("PromotionsLast3Yrs")["AttritionBinary"].mean()
    ax.bar(promo_rate.index.astype(str), promo_rate.values, color="#7c6af7", alpha=0.9)
    for x, v in enumerate(promo_rate.values):
        ax.text(x, v+0.003, f"{v:.1%}", ha="center", color=FG, fontsize=9)
    ax.set_xlabel("Promotions in Last 3 Years", color=FG)
    ax.set_title("By Promotion Count", color=FG)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_band_sat.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/eda_band_sat.png")


def plot_eda_pay_wfh(df: pd.DataFrame, out_dir: str) -> None:
    """Pay vs market distribution and WFH flexibility curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]; ax_style(ax, fig)
    for lbl, g in df.groupby("Attrition"):
        ax.hist(g["MarketRateRatio"], bins=40, alpha=0.65, density=True,
                label=lbl, color={"Yes":"#ef4444","No":"#10b981"}[lbl])
    ax.axvline(1.0, color="white", ls="--", lw=1.5, label="Market Parity")
    ax.set_xlabel("Market Rate Ratio  (<1 = underpaid)", color=FG)
    ax.set_ylabel("Density", color=FG)
    ax.set_title("Pay vs Market — Attrition vs Retained", color=FG)
    ax.legend(facecolor=PANEL, labelcolor=FG)

    ax = axes[1]; ax_style(ax, fig)
    wfh_rate = df.groupby("WFH_DaysPerWeek")["AttritionBinary"].mean()
    ax.plot(wfh_rate.index, wfh_rate.values, "o-", color="#10b981", lw=2.5, ms=8)
    ax.fill_between(wfh_rate.index, wfh_rate.values, alpha=0.15, color="#10b981")
    for x, y in zip(wfh_rate.index, wfh_rate.values):
        ax.text(x, y+0.005, f"{y:.1%}", ha="center", color=FG, fontsize=8)
    ax.set_xlabel("WFH Days per Week", color=FG)
    ax.set_ylabel("Attrition Rate", color=FG)
    ax.set_title("Attrition Rate vs WFH Flexibility", color=FG)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_pay_wfh.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/eda_pay_wfh.png")


def plot_eda_heatmaps(df: pd.DataFrame, out_dir: str) -> None:
    """Gender × Dept and LevelBand × Shift heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    pivot1 = df.pivot_table(values="AttritionBinary",
                             index="Department", columns="Gender",
                             aggfunc="mean") * 100
    sns.heatmap(pivot1, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=axes[0], linewidths=0.5, annot_kws={"size":9})
    axes[0].set_title("Attrition % — Gender × Department", color=FG)
    axes[0].tick_params(colors=FG)

    pivot2 = df.pivot_table(values="AttritionBinary",
                             index="LevelBand", columns="ShiftType",
                             aggfunc="mean") * 100
    pivot2 = pivot2.reindex([b for b in BAND_ORDER if b in pivot2.index])
    sns.heatmap(pivot2, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=axes[1], linewidths=0.5, annot_kws={"size":9})
    axes[1].set_title("Attrition % — Level Band × Shift Type", color=FG)
    axes[1].tick_params(colors=FG)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_heatmaps.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/eda_heatmaps.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MODEL COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(results: dict, y_te, out_dir: str) -> None:
    """ROC curve for each model."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax_style(ax, fig)

    for (name, res), c in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f"{name.split()[0]} (AUC={res['test_auc']:.3f})")
    ax.plot([0,1],[0,1], "w--", lw=0.8, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", color=FG)
    ax.set_ylabel("True Positive Rate", color=FG)
    ax.set_title("ROC Curves — Test Set", color=FG, fontsize=12)
    ax.legend(facecolor=PANEL, labelcolor=FG, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/roc_curves.png")


def plot_model_comparison(results: dict, out_dir: str) -> None:
    """Grouped bar chart comparing CV AUC, Test AUC, F1 and Recall."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax_style(ax, fig)

    names = list(results.keys())
    x, w  = np.arange(len(names)), 0.20
    ax.bar(x-1.5*w, [results[n]["cv_auc_mean"]  for n in names], w,
           color="#7c6af7", label="CV AUC",   alpha=0.9)
    ax.bar(x-0.5*w, [results[n]["test_auc"]      for n in names], w,
           color="#10b981", label="Test AUC", alpha=0.9)
    ax.bar(x+0.5*w, [results[n]["test_f1"]       for n in names], w,
           color="#f59e0b", label="F1",       alpha=0.9)
    ax.bar(x+1.5*w, [results[n]["test_recall"]   for n in names], w,
           color="#ef4444", label="Recall",   alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ","\n") for n in names], color=FG, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison", color=FG, fontsize=12)
    ax.legend(facecolor=PANEL, labelcolor=FG)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/model_comparison.png")


def plot_confusion_matrix(results: dict, best_name: str,
                           y_te, out_dir: str) -> None:
    """Confusion matrix for the best model."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax_style(ax, fig)
    cm = confusion_matrix(y_te, results[best_name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stay","Leave"], yticklabels=["Stay","Leave"],
                annot_kws={"size":14})
    ax.set_title(f"Confusion Matrix — {best_name}", color=FG)
    ax.set_xlabel("Predicted", color=FG); ax.set_ylabel("Actual", color=FG)
    ax.tick_params(colors=FG)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FEATURE IMPORTANCE PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(best_model, feature_names: list,
                             best_name: str, X_te, y_te,
                             needs_scale: bool, scaler,
                             out_dir: str) -> None:
    """Bar chart of top 25 feature importances."""
    Xte_imp = scaler.transform(X_te) if needs_scale else X_te

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        r = permutation_importance(best_model, Xte_imp, y_te,
                                   n_repeats=10, random_state=42)
        importances = r.importances_mean

    imp_df = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(25))

    print(f"\n  Top 10 Features ({best_name}):")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['Feature']:30s}  {row['Importance']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 9))
    ax_style(ax, fig)
    colors_imp = plt.cm.RdYlGn_r(np.linspace(0.1, 0.85, len(imp_df)))
    ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
            color=colors_imp[::-1], alpha=0.9)
    ax.set_title(f"Top 25 Feature Importances — {best_name}", color=FG, fontsize=12)
    ax.set_xlabel("Importance", color=FG)
    ax.tick_params(colors=FG, labelsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(df_scored: pd.DataFrame, df_raw: pd.DataFrame,
                   best_model, scaler, encoders: dict,
                   results: dict, best_name: str,
                   feature_names: list, out_dir: str) -> None:
    """
    Persist all artifacts needed by the Streamlit app.
      artifacts/
        best_model.pkl
        scaler.pkl
        encoders.pkl
        model_meta.json
        Microland_employees.csv
        employees_scored.csv
    """
    # Remove stale folder — avoids Windows permission errors on locked files
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # CSVs
    df_raw.to_csv(f"{out_dir}/Microland_employees.csv", index=False)
    df_scored.to_csv(f"{out_dir}/employees_scored.csv", index=False)
    print(f"  Saved → {out_dir}/Microland_employees.csv")
    print(f"  Saved → {out_dir}/employees_scored.csv")

    # Model artifacts
    pickle.dump(best_model, open(f"{out_dir}/best_model.pkl", "wb"))
    pickle.dump(scaler,     open(f"{out_dir}/scaler.pkl",     "wb"))
    pickle.dump(encoders,   open(f"{out_dir}/encoders.pkl",   "wb"))
    print(f"  Saved → {out_dir}/best_model.pkl  scaler.pkl  encoders.pkl")

    # Metadata JSON
    r    = results[best_name]
    meta = {
        "best_model_name"    : best_name,
        "needs_scale"        : r["needs_scale"],
        "feature_names"      : feature_names,
        "drop_cols"          : DROP_COLS,
        "categorical_cols"   : CATEGORICAL,
        "decision_threshold" : DECISION_THRESHOLD,
        "risk_thresholds"    : {"Critical": 0.45, "High": 0.32, "Medium": 0.18},
        "test_auc"           : round(r["test_auc"],    4),
        "test_f1"            : round(r["test_f1"],     4),
        "test_acc"           : round(r["test_acc"],    4),
        "test_recall"        : round(r["test_recall"], 4),
        "cv_auc"             : round(r["cv_auc_mean"], 4),
        "brier_score"        : round(r.get("brier", 0.0), 4),
        "is_calibrated"      : bool(r.get("is_calibrated", False)),
        "all_models": {
            n: {
                "cv_auc"      : round(v["cv_auc_mean"], 4),
                "test_auc"    : round(v["test_auc"],    4),
                "test_f1"     : round(v["test_f1"],     4),
                "test_recall" : round(v["test_recall"], 4),
                "test_acc"    : round(v["test_acc"],    4),
                "brier_score" : round(v.get("brier", 0.0), 4),
            } for n, v in results.items()
        },
        "dataset_size" : len(df_scored),
        "trained_on"   : datetime.now().isoformat(),
    }
    json.dump(meta, open(f"{out_dir}/model_meta.json", "w"), indent=2)
    print(f"  Saved → {out_dir}/model_meta.json")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  🏢  Microland Attrition — Training Pipeline")
    print("═" * 60)

    # ── Step 1: Generate data ─────────────────────────────────────
    print("\n[1/7] Generating 5000-employee dataset...")
    df = generate_microland_employees(N=5000, seed=42)
    print(f"      Shape: {df.shape} | Attrition rate: {df['AttritionBinary'].mean():.1%}")

    # ── Step 2: Validate personas ─────────────────────────────────
    print("\n[2/7] Validating attrition personas...")
    validate_personas(df)

    # ── Step 3: Feature engineering ───────────────────────────────
    print("\n[3/7] Engineering features...")
    df = engineer_features(df)
    print(f"      Total features: {df.shape[1]}")

    # ── Step 4: EDA plots ─────────────────────────────────────────
    print(f"\n[4/7] Generating EDA plots → {OUT}/")
    os.makedirs(OUT, exist_ok=True)
    plot_eda_overview(df, OUT)
    plot_eda_categorical(df, OUT)
    plot_eda_band_sat(df, OUT)
    plot_eda_pay_wfh(df, OUT)
    plot_eda_heatmaps(df, OUT)

    # ── Step 5: Preprocessing ─────────────────────────────────────
    print("\n[5/7] Preprocessing...")
    X, y, encoders = preprocess(df, fit=True)
    feature_names  = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"      Train: {len(X_tr):,} | Test: {len(X_te):,} | Features: {len(feature_names)}")

    # ── Step 6: Train & compare models ───────────────────────────
    print("\n[6/7] Training & evaluating models...")
    results     = train_all_models(X_tr, X_te, X_tr_sc, X_te_sc, y_tr, y_te, cv)
    best_name   = max(results, key=lambda k: results[k]["cv_auc_mean"])
    best_model  = results[best_name]["model"]
    needs_scale = results[best_name]["needs_scale"]

    print(f"\n  🏆  Best model: {best_name}  "
          f"(CV AUC={results[best_name]['cv_auc_mean']:.4f}  "
          f"Test AUC={results[best_name]['test_auc']:.4f})")

    print(f"\n  Classification Report — {best_name}")
    print("  " + "-" * 50)
    print(classification_report(
        y_te, results[best_name]["y_pred"],
        target_names=["Stay","Leave"]
    ))

    # Model comparison plots
    plot_roc_curves(results, y_te, OUT)
    plot_model_comparison(results, OUT)
    plot_confusion_matrix(results, best_name, y_te, OUT)
    plot_feature_importance(best_model, feature_names, best_name,
                            X_te, y_te, needs_scale, scaler, OUT)

    # ── Step 7: Score all employees & save ───────────────────────
    print(f"\n[7/7] Scoring all employees & saving artifacts → {OUT}/")
    X_full   = X.values
    X_pred   = scaler.transform(X_full) if needs_scale else X_full
    df["AttritionProb"] = best_model.predict_proba(X_pred)[:, 1].round(4)
    df["RiskCategory"]  = df["AttritionProb"].apply(risk_cat)
    df_scored           = df.copy()

    print(f"\n  Risk category distribution:")
    print(df["RiskCategory"].value_counts().to_string())

    # Final persona check using model predictions
    ideal = df_scored[
        (df_scored["Gender"]           == "Female")  &
        (df_scored["MaritalStatus"]    == "Married") &
        (df_scored["Location"]         == "WFH")     &
        (df_scored["PromotionsLast3Yrs"] >= 1)       &
        (df_scored["AwardsReceived"]   >= 1)         &
        (df_scored["JobSatisfaction"]  >= 3)         &
        (df_scored["OverTime"]         == "No")
    ]
    risky = df_scored[
        (df_scored["MaritalStatus"]   == "Single")                    &
        (df_scored["Department"]      == "Sales")                     &
        (df_scored["MarketRateRatio"] < 0.85)                         &
        (df_scored["OverTime"]        == "Yes")                       &
        (df_scored["ShiftType"].isin(["Night","Rotational"]))
    ]
    print(f"\n  💚 Married WFH Woman, Promoted → {ideal['AttritionProb'].mean():.1%}")
    print(f"  🔴 Single Sales Underpaid OT   → {risky['AttritionProb'].mean():.1%}")
    print(f"  📊 Company Average             → {df_scored['AttritionProb'].mean():.1%}")

    save_artifacts(df_scored, df, best_model, scaler, encoders,
                   results, best_name, feature_names, OUT)

    # ── Final summary ─────────────────────────────────────────────
    r = results[best_name]
    print("\n" + "═" * 60)
    print(f"  🏆  DEPLOYED  :  {best_name}")
    print("═" * 60)
    print(f"  Test AUC    :  {r['test_auc']:.4f}")
    print(f"  Brier Score :  {r.get('brier', 0.0):.4f}")
    print(f"  Test F1     :  {r['test_f1']:.4f}")
    print(f"  Recall      :  {r['test_recall']:.4f}")
    print(f"  Dataset     :  {len(df_scored):,} employees")
    print(f"  Features    :  {len(feature_names)}")
    print("═" * 60)
    print("\n  ▶  streamlit run app.py\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
