"""
TechNova Inc. – Employee Attrition Prediction Dashboard
Streamlit App
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="TechNova – Attrition Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0f1117; }
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #252840 100%);
        border: 1px solid #3a3d5c;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .risk-critical { background: linear-gradient(135deg, #7f1d1d, #991b1b); border: 1px solid #dc2626; border-radius: 8px; padding: 10px; }
    .risk-high     { background: linear-gradient(135deg, #7c2d12, #9a3412); border: 1px solid #ea580c; border-radius: 8px; padding: 10px; }
    .risk-medium   { background: linear-gradient(135deg, #713f12, #854d0e); border: 1px solid #ca8a04; border-radius: 8px; padding: 10px; }
    .risk-low      { background: linear-gradient(135deg, #14532d, #166534); border: 1px solid #16a34a; border-radius: 8px; padding: 10px; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data Generation (cached) ─────────────────────────────────
@st.cache_data
def generate_data(N=2000, seed=42):
    np.random.seed(seed)
    EDUCATION_LEVELS = {1:"No Formal Qualifications",2:"High School",3:"Bachelor",4:"Master",5:"Doctorate"}
    levels=[\"L1\",\"L2\",\"L3\",\"L4\",\"L5\",\"L6\"]; level_probs=[0.35,0.28,0.18,0.11,0.06,0.02]
    level_salary={\"L1\":400000,\"L2\":650000,\"L3\":950000,\"L4\":1400000,\"L5\":2000000,\"L6\":3000000}
    dept_mul={\"Engineering\":1.20,\"Finance\":1.10,\"Legal\":1.15,\"Sales\":0.95,\"Marketing\":0.95,\"HR\":0.90,\"Operations\":0.85}
    depts=[\"Engineering\",\"Sales\",\"HR\",\"Finance\",\"Marketing\",\"Operations\",\"Legal\"]
    dept_w=[0.35,0.20,0.08,0.10,0.10,0.12,0.05]
    locs=[\"Bengaluru\",\"Mumbai\",\"Hyderabad\",\"Pune\",\"Chennai\",\"Delhi\",\"Remote\"]
    loc_p=[0.30,0.20,0.15,0.12,0.10,0.08,0.05]
    first_names=[\"Aarav\",\"Priya\",\"Rohan\",\"Sneha\",\"Vikram\",\"Ananya\",\"Arjun\",\"Kavya\",\"Dev\",\"Meera\"]
    last_names=[\"Sharma\",\"Patel\",\"Kumar\",\"Singh\",\"Mehta\",\"Gupta\",\"Joshi\",\"Nair\",\"Rao\",\"Iyer\"]

    hire_start=datetime(2010,1,1); ref_date=datetime(2024,6,30)
    hire_dates=[hire_start+timedelta(days=int(x)) for x in np.random.uniform(0,(datetime(2023,12,31)-hire_start).days,N)]
    yac=[(ref_date-h).days/365 for h in hire_dates]

    level=np.random.choice(levels,N,p=level_probs)
    loc=np.random.choice(locs,N,p=loc_p)
    dept=np.random.choice(depts,N,p=dept_w)
    gender=np.random.choice([\"Male\",\"Female\",\"Non-Binary\"],N,p=[0.52,0.44,0.04])
    age=np.clip(np.random.normal(35,9,N).astype(int),22,60)
    edu_id=np.random.choice([1,2,3,4,5],N,p=[0.03,0.10,0.45,0.32,0.10])
    marital=np.random.choice([\"Single\",\"Married\",\"Divorced\"],N,p=[0.35,0.50,0.15])
    overtime=np.random.choice([\"Yes\",\"No\"],N,p=[0.28,0.72])
    biz_travel=np.random.choice([\"None\",\"Rarely\",\"Frequently\"],N,p=[0.30,0.45,0.25])
    dist_km=np.clip(np.random.exponential(15,N).astype(int),1,100)
    stock_opt=np.random.choice([0,1,2,3],N,p=[0.35,0.35,0.20,0.10])
    salary=np.array([int(level_salary[l]*dept_mul[d]*np.random.uniform(0.85,1.15)) for l,d in zip(level,dept)])
    mkt=np.clip(np.random.normal(1.0,0.15,N),0.60,1.50)
    yrs_promo=np.array([min(y*np.random.uniform(0,0.8),y) for y in yac])
    yrs_role =np.array([min(y*np.random.uniform(0.2,1),y) for y in yac])
    yrs_mgr  =np.array([min(y*np.random.uniform(0.1,0.9),y) for y in yac])
    l30=np.random.poisson(0.8,N); l90=l30+np.random.poisson(1.5,N)
    l180=l90+np.random.poisson(2,N); l365=l180+np.random.poisson(3,N)
    awards=np.random.choice([0,1,2,3,4],N,p=[0.45,0.30,0.15,0.07,0.03])
    env_s=np.random.randint(1,6,N); job_s=np.random.randint(1,6,N)
    rel_s=np.random.randint(1,6,N); wlb=np.random.randint(1,6,N)
    srate=np.random.randint(1,6,N); mrate=np.clip(srate+np.random.randint(-1,2,N),1,5)
    tr_av=np.random.randint(2,8,N); tr_tk=np.array([np.random.randint(0,t+1) for t in tr_av])

    risk=(-1.5*mkt+0.8*(overtime==\"Yes\").astype(int)-0.5*(job_s/5)-0.4*(env_s/5)-0.3*(wlb/5)
          +0.6*(yrs_promo/(np.array(yac)+0.1))+0.4*(l90/5)-0.3*(awards/4)
          +0.3*(biz_travel==\"Frequently\").astype(int)+0.2*(dist_km/100)+np.random.normal(0,0.5,N))
    attr_prob=1/(1+np.exp(-risk))
    attr=(np.random.uniform(0,1,N)<attr_prob).astype(int)

    emp_ids=[f\"EMP{str(i+1).zfill(5)}\" for i in range(N)]
    df=pd.DataFrame({
        \"EmployeeID\":emp_ids, \"FirstName\":np.random.choice(first_names,N),
        \"LastName\":np.random.choice(last_names,N), \"Gender\":gender, \"Age\":age,
        \"Level\":level, \"Location\":loc, \"Department\":dept,
        \"BusinessTravel\":biz_travel, \"DistanceFromHome_KM\":dist_km,
        \"MaritalStatus\":marital, \"Salary\":salary, \"MarketRateRatio\":mkt.round(3),
        \"StockOptionLevel\":stock_opt, \"OverTime\":overtime,
        \"YearsAtCompany\":np.round(yac,2), \"YearsInMostRecentRole\":np.round(yrs_role,2),
        \"YearsSinceLastPromotion\":np.round(yrs_promo,2), \"YearsWithCurrManager\":np.round(yrs_mgr,2),
        \"EducationLevelID\":edu_id, \"EducationLevel\":[EDUCATION_LEVELS[i] for i in edu_id],
        \"Leave_Last30Days\":l30, \"Leave_Last90Days\":l90,
        \"Leave_Last180Days\":l180, \"Leave_Last365Days\":l365,
        \"AwardsReceived\":awards, \"Attrition\":[\"Yes\" if a else \"No\" for a in attr],
        \"EnvironmentSatisfaction\":env_s, \"JobSatisfaction\":job_s,
        \"RelationshipSatisfaction\":rel_s, \"WorkLifeBalance\":wlb,
        \"SelfRating\":srate, \"ManagerRating\":mrate,
        \"TrainingOpportunitiesWithinYear\":tr_av, \"TrainingOpportunitiesTaken\":tr_tk,
    })
    # Feature engineering
    df[\"TrainingUtilizationRate\"]=df[\"TrainingOpportunitiesTaken\"]/df[\"TrainingOpportunitiesWithinYear\"].clip(lower=1)
    df[\"AvgSatisfaction\"]=df[[\"EnvironmentSatisfaction\",\"JobSatisfaction\",\"RelationshipSatisfaction\",\"WorkLifeBalance\"]].mean(axis=1)
    df[\"RatingGap\"]=df[\"ManagerRating\"]-df[\"SelfRating\"]
    df[\"PromotionStagnation\"]=df[\"YearsSinceLastPromotion\"]/(df[\"YearsAtCompany\"]+0.1)
    df[\"LeaveIntensity\"]=df[\"Leave_Last90Days\"]/90
    df[\"PayDeficit\"]=(1-df[\"MarketRateRatio\"]).clip(lower=0)
    df[\"IsNewbie\"]=(df[\"YearsAtCompany\"]<1).astype(int)
    df[\"IsVeteran\"]=(df[\"YearsAtCompany\"]>10).astype(int)
    df[\"OverTimeFlag\"]=(df[\"OverTime\"]==\"Yes\").astype(int)
    return df

@st.cache_resource
def train_all_models(df):
    target=\"Attrition\"
    drop_cols=[\"EmployeeID\",\"FirstName\",\"LastName\",\"Attrition\",\"EducationLevel\"]
    feature_cols=[c for c in df.columns if c not in drop_cols]
    X=df[feature_cols].copy()
    y=(df[target]==\"Yes\").astype(int)
    for c in X.select_dtypes(include=[\"object\",\"category\"]).columns:
        X[c]=LabelEncoder().fit_transform(X[c].astype(str))
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler=StandardScaler()
    X_trs=scaler.fit_transform(X_tr); X_tes=scaler.transform(X_te)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    models={
        \"Logistic Regression\":LogisticRegression(max_iter=1000,class_weight=\"balanced\",C=0.5),
        \"Decision Tree\":DecisionTreeClassifier(max_depth=6,class_weight=\"balanced\",random_state=42),
        \"Random Forest\":RandomForestClassifier(n_estimators=100,max_depth=8,class_weight=\"balanced\",random_state=42),
        \"Gradient Boosting\":GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,max_depth=4,random_state=42),
        \"K-Nearest Neighbors\":KNeighborsClassifier(n_neighbors=9),
        \"SVM (RBF)\":SVC(kernel=\"rbf\",probability=True,class_weight=\"balanced\",C=1.0),
    }
    results={}
    for name,model in models.items():
        sc=(name in [\"Logistic Regression\",\"K-Nearest Neighbors\",\"SVM (RBF)\"])
        Xtr,Xte=(X_trs,X_tes) if sc else (X_tr.values,X_te.values)
        cv_auc=cross_val_score(model,Xtr,y_tr,cv=cv,scoring=\"roc_auc\")
        model.fit(Xtr,y_tr)
        yp=model.predict(Xte); ypr=model.predict_proba(Xte)[:,1]
        results[name]={\"model\":model,\"scaled\":sc,\"y_pred\":yp,\"y_prob\":ypr,
                        \"cv_auc\":cv_auc.mean(),\"cv_auc_std\":cv_auc.std(),
                        \"test_auc\":roc_auc_score(y_te,ypr),\"test_f1\":f1_score(y_te,yp),
                        \"test_acc\":accuracy_score(y_te,yp)}
    best_name=max(results,key=lambda k:results[k][\"test_auc\"])
    best=results[best_name]
    Xall=scaler.transform(X) if best[\"scaled\"] else X.values
    df2=df.copy()
    df2[\"AttritionProb\"]=best[\"model\"].predict_proba(Xall)[:,1]
    df2[\"RiskCategory\"]=pd.cut(df2[\"AttritionProb\"],bins=[0,.25,.5,.75,1.0],
                                  labels=[\"Low\",\"Medium\",\"High\",\"Critical\"])
    return df2,results,best_name,X,y,y_te,scaler,feature_cols

# ── Load data ────────────────────────────────────────────────
with st.spinner("🔄 Generating data & training models..."):
    df_raw = generate_data()
    df_scored, results, best_name, X_full, y_full, y_test, scaler, feature_cols = train_all_models(df_raw)

best = results[best_name]

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60/1a1d2e/7c6af7?text=TechNova+Inc.", width=200)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Overview","🔍 EDA","🤖 Models","🎯 Risk Engine","👤 Employee Lookup"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Best Model:** {best_name}")
st.sidebar.markdown(f"**Test AUC:** {best['test_auc']:.3f}")
st.sidebar.markdown(f"**F1 Score:** {best['test_f1']:.3f}")
st.sidebar.markdown(f"**Dataset:** {len(df_raw):,} employees")
st.sidebar.markdown(f"**Attrition Rate:** {(df_raw['Attrition']=='Yes').mean():.1%}")

# ── Overview Page ────────────────────────────────────────────
if page == "📊 Overview":
    st.title("🏢 TechNova Inc. – Attrition Prediction Dashboard")

    c1,c2,c3,c4,c5 = st.columns(5)
    total = len(df_scored)
    attr_n = (df_raw["Attrition"]=="Yes").sum()
    critical = (df_scored["RiskCategory"]=="Critical").sum()
    high_risk = (df_scored["RiskCategory"].isin(["High","Critical"])).sum()

    c1.metric("Total Employees", f"{total:,}")
    c2.metric("Historical Attritions", f"{attr_n}", delta=f"{attr_n/total:.1%}")
    c3.metric("Critical Risk", f"{critical}", delta="Immediate attention")
    c4.metric("High+ Risk", f"{high_risk}", delta=f"{high_risk/total:.1%}")
    c5.metric("Best Model AUC", f"{best['test_auc']:.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution by Level (Pyramid)")
        levels_order = ["L1","L2","L3","L4","L5","L6"]
        lv = df_scored.groupby("Level")["AttritionProb"].mean().reindex(levels_order)
        fig,ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        colors_grad = plt.cm.RdYlGn_r(np.linspace(0.2,0.8,6))
        bars = ax.barh(lv.index, lv.values, color=colors_grad)
        for b,v in zip(bars,lv.values):
            ax.text(v+0.002, b.get_y()+b.get_height()/2, f"{v:.1%}", va="center", color="white", fontsize=9)
        ax.tick_params(colors="white"); ax.set_xlabel("Avg Attrition Probability", color="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Attrition Risk by Department")
        dp = df_scored.groupby("Department")["AttritionProb"].mean().sort_values(ascending=True)
        fig,ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        ax.barh(dp.index, dp.values, color="#f59e0b", alpha=0.85)
        for i,(idx,v) in enumerate(dp.items()):
            ax.text(v+0.002, i, f"{v:.1%}", va="center", color="white", fontsize=9)
        ax.tick_params(colors="white"); ax.set_xlabel("Avg Attrition Probability", color="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        st.pyplot(fig); plt.close()

    col3,col4 = st.columns(2)
    with col3:
        st.subheader("Risk Category Breakdown")
        rc = df_scored["RiskCategory"].value_counts().sort_index()
        fig,ax = plt.subplots(figsize=(6,4)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        c_map = {"Low":"#22c55e","Medium":"#eab308","High":"#f97316","Critical":"#ef4444"}
        ax.pie(rc.values, labels=rc.index, colors=[c_map.get(l,"#888") for l in rc.index],
               autopct="%1.1f%%", textprops={"color":"white"}, startangle=90)
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Attrition Risk by Location")
        lc = df_scored.groupby("Location")["AttritionProb"].mean().sort_values(ascending=False)
        fig,ax = plt.subplots(figsize=(6,4)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        ax.bar(lc.index, lc.values, color="#10b981", alpha=0.85)
        ax.tick_params(axis='x', rotation=30, colors="white"); ax.tick_params(axis='y', colors="white")
        ax.set_ylabel("Avg Risk Score", color="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        st.pyplot(fig); plt.close()

# ── EDA Page ─────────────────────────────────────────────────
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    dim = st.selectbox("Analyze attrition by:", ["Gender","Department","Level","Location",
                                                   "MaritalStatus","OverTime","BusinessTravel",
                                                   "JobSatisfaction","ManagerRating","AwardsReceived"])
    col1,col2 = st.columns(2)
    with col1:
        fig,ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        grp = df_raw.groupby(dim)["Attrition"].apply(lambda x:(x=="Yes").mean()).sort_values(ascending=False)
        ax.bar(grp.index.astype(str), grp.values, color="#7c6af7", alpha=0.85)
        ax.set_ylabel("Attrition Rate", color="white"); ax.set_title(f"Attrition Rate by {dim}", color="white")
        ax.tick_params(axis='x', rotation=30, colors="white"); ax.tick_params(axis='y', colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        st.pyplot(fig); plt.close()

    with col2:
        fig,ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        for lbl,grp in df_raw.groupby("Attrition"):
            ax.hist(grp["MarketRateRatio"], bins=30, alpha=0.6, label=lbl,
                    color={"Yes":"#ff4b4b","No":"#00c9b1"}[lbl])
        ax.axvline(1.0,color="white",ls="--",lw=1,label="Market Parity")
        ax.set_xlabel("Market Rate Ratio", color="white"); ax.set_ylabel("Count", color="white")
        ax.set_title("Pay vs Market by Attrition", color="white")
        ax.legend(facecolor="#1a1d2e",labelcolor="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        st.pyplot(fig); plt.close()

    st.subheader("Summary Statistics")
    num_summary = df_raw[["Age","Salary","MarketRateRatio","YearsAtCompany",
                           "YearsSinceLastPromotion","Leave_Last90Days","AwardsReceived"]].describe().round(2)
    st.dataframe(num_summary, use_container_width=True)

# ── Models Page ──────────────────────────────────────────────
elif page == "🤖 Models":
    st.title("🤖 Model Comparison & Analysis")

    # Metrics table
    metrics_df = pd.DataFrame({
        "Model": list(results.keys()),
        "CV AUC": [f"{results[n]['cv_auc']:.3f} ±{results[n]['cv_auc_std']:.3f}" for n in results],
        "Test AUC": [f"{results[n]['test_auc']:.3f}" for n in results],
        "F1 Score": [f"{results[n]['test_f1']:.3f}" for n in results],
        "Accuracy": [f"{results[n]['test_acc']:.3f}" for n in results],
        "Best": ["✅" if n==best_name else "" for n in results],
    })
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        fig,ax = plt.subplots(figsize=(7,5)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        colors6=["#7c6af7","#f59e0b","#10b981","#f97316","#ec4899","#06b6d4"]
        for (n,res),c in zip(results.items(),colors6):
            fpr,tpr,_=roc_curve(y_test,res["y_prob"])
            ax.plot(fpr,tpr,color=c,lw=1.5,label=f"{n.split()[0]} ({res['test_auc']:.3f})")
        ax.plot([0,1],[0,1],"w--",lw=0.8,alpha=0.4)
        ax.set_xlabel("FPR",color="white"); ax.set_ylabel("TPR",color="white")
        ax.set_title("ROC Curves",color="white")
        ax.legend(facecolor="#1a1d2e",labelcolor="white",fontsize=8)
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader(f"Confusion Matrix – {best_name}")
        from sklearn.metrics import confusion_matrix
        y_te_arr = (df_raw["Attrition"]=="Yes").values
        # use test predictions
        cm=confusion_matrix(y_test, best["y_pred"])
        fig,ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        ax.imshow(cm,cmap="magma")
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=16,fontweight="bold",
                        color="white" if cm[i,j]<cm.max()*0.7 else "black")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Stay","Leave"],color="white"); ax.set_yticklabels(["Stay","Leave"],color="white")
        ax.set_title("Confusion Matrix",color="white")
        st.pyplot(fig); plt.close()

    st.subheader(f"Feature Importance – {best_name}")
    bm=best["model"]; fn=list(X_full.columns)
    fig,ax = plt.subplots(figsize=(14,5)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
    if hasattr(bm,"feature_importances_"):
        imp=bm.feature_importances_; si=np.argsort(imp)[-20:]
        ax.barh([fn[i] for i in si],imp[si],color="#7c6af7",alpha=0.85)
        ax.set_title("Top 20 Feature Importances",color="white")
    elif hasattr(bm,"coef_"):
        coef=np.abs(bm.coef_[0]); si=np.argsort(coef)[-20:]
        ax.barh([fn[i] for i in si],coef[si],color="#f59e0b",alpha=0.85)
        ax.set_title("Top 20 Coefficient Magnitudes",color="white")
    else:
        ax.text(0.5,0.5,"Feature importance not available for this model",
                ha="center",va="center",color="white",transform=ax.transAxes)
    ax.tick_params(colors="white",labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#444466")
    st.pyplot(fig); plt.close()

# ── Risk Engine Page ─────────────────────────────────────────
elif page == "🎯 Risk Engine":
    st.title("🎯 Employee Risk Prediction Engine")
    st.markdown("Enter employee details to get an **instant attrition probability score**.")

    with st.form("predict_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age_i        = st.slider("Age", 22, 60, 32)
            level_i      = st.selectbox("Level", ["L1","L2","L3","L4","L5","L6"])
            dept_i       = st.selectbox("Department", ["Engineering","Sales","HR","Finance","Marketing","Operations","Legal"])
            gender_i     = st.selectbox("Gender", ["Male","Female","Non-Binary"])
            location_i   = st.selectbox("Location", ["Bengaluru","Mumbai","Hyderabad","Pune","Chennai","Delhi","Remote"])
        with c2:
            overtime_i   = st.selectbox("OverTime", ["No","Yes"])
            biz_travel_i = st.selectbox("Business Travel", ["None","Rarely","Frequently"])
            marital_i    = st.selectbox("Marital Status", ["Single","Married","Divorced"])
            stock_i      = st.selectbox("Stock Option Level", [0,1,2,3])
            edu_i        = st.selectbox("Education Level", ["Bachelor","Master","High School","Doctorate","No Formal Qualifications"])
        with c3:
            mkt_rate_i   = st.slider("Market Rate Ratio", 0.60, 1.50, 1.0, 0.01)
            job_sat_i    = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
            wlb_i        = st.slider("Work-Life Balance (1-5)", 1, 5, 3)
            awards_i     = st.slider("Awards Received", 0, 4, 0)
            yrs_promo_i  = st.slider("Years Since Last Promotion", 0.0, 15.0, 2.0, 0.5)
            l90_i        = st.slider("Leaves (Last 90 Days)", 0, 15, 2)
        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)

    if submitted:
        # Build a single-row dataframe matching training features
        edu_map={"No Formal Qualifications":1,"High School":2,"Bachelor":3,"Master":4,"Doctorate":5}
        sample = pd.DataFrame([{
            "Gender":gender_i, "Age":age_i, "Level":level_i, "Location":location_i,
            "Department":dept_i, "BusinessTravel":biz_travel_i, "DistanceFromHome_KM":15,
            "MaritalStatus":marital_i, "Salary":700000, "MarketRateRatio":mkt_rate_i,
            "StockOptionLevel":stock_i, "OverTime":overtime_i, "YearsAtCompany":3.0,
            "YearsInMostRecentRole":2.0, "YearsSinceLastPromotion":yrs_promo_i,
            "YearsWithCurrManager":2.0, "EducationLevelID":edu_map[edu_i],
            "Leave_Last30Days":int(l90_i*0.3), "Leave_Last90Days":l90_i,
            "Leave_Last180Days":int(l90_i*1.8), "Leave_Last365Days":int(l90_i*3),
            "AwardsReceived":awards_i, "EnvironmentSatisfaction":3, "JobSatisfaction":job_sat_i,
            "RelationshipSatisfaction":3, "WorkLifeBalance":wlb_i, "SelfRating":3,
            "ManagerRating":3, "TrainingOpportunitiesWithinYear":4, "TrainingOpportunitiesTaken":2,
            "TrainingUtilizationRate":0.5, "AvgSatisfaction":job_sat_i,
            "RatingGap":0, "PromotionStagnation":yrs_promo_i/3.1,
            "LeaveIntensity":l90_i/90, "PayDeficit":max(0,1-mkt_rate_i),
            "IsNewbie":0, "IsVeteran":0, "OverTimeFlag":int(overtime_i=="Yes"),
        }])
        # Align to feature_cols
        for col in feature_cols:
            if col not in sample.columns:
                sample[col] = 0
        sample = sample[feature_cols]
        for c in sample.select_dtypes(include=["object","category"]).columns:
            le = LabelEncoder()
            le.fit(df_raw[c].astype(str).unique() if c in df_raw.columns else [str(sample[c].iloc[0])])
            try: sample[c] = le.transform(sample[c].astype(str))
            except: sample[c] = 0

        bm=best["model"]; sc=best["scaled"]
        X_s = scaler.transform(sample) if sc else sample.values
        prob = bm.predict_proba(X_s)[0,1]

        st.markdown("---")
        cat = "Critical 🔴" if prob>0.75 else "High 🟠" if prob>0.5 else "Medium 🟡" if prob>0.25 else "Low 🟢"
        col1,col2,col3 = st.columns(3)
        col1.metric("Attrition Probability", f"{prob:.1%}")
        col2.metric("Risk Category", cat)
        col3.metric("Model Used", best_name)

        fig,ax = plt.subplots(figsize=(8,2)); fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d2e')
        cmap = plt.cm.RdYlGn_r
        gradient = np.linspace(0,1,300).reshape(1,-1)
        ax.imshow(gradient, cmap=cmap, aspect="auto", extent=[0,1,0,1])
        ax.axvline(prob, color="white", lw=3)
        ax.text(prob, 0.5, f"  {prob:.1%}", color="white", va="center", fontsize=14, fontweight="bold")
        ax.set_xlim(0,1); ax.set_yticks([]); ax.set_xlabel("Attrition Risk →", color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        st.pyplot(fig); plt.close()

        st.subheader("Key Risk Drivers")
        drivers = []
        if prob > 0.5: drivers.append(f"⚠️ Market Rate Ratio {mkt_rate_i:.2f} — {'below parity' if mkt_rate_i<1 else 'at/above parity'}")
        if overtime_i=="Yes": drivers.append("⚠️ Employee works overtime")
        if job_sat_i<=2: drivers.append("⚠️ Low job satisfaction")
        if yrs_promo_i>4: drivers.append("⚠️ Over 4 years without promotion")
        if l90_i>5: drivers.append("⚠️ High absenteeism (90-day leaves)")
        if awards_i==0: drivers.append("⚠️ No awards received")
        if biz_travel_i=="Frequently": drivers.append("⚠️ Frequent business travel")
        if not drivers: drivers.append("✅ No major red flags detected")
        for d in drivers: st.markdown(d)

# ── Employee Lookup ──────────────────────────────────────────
elif page == "👤 Employee Lookup":
    st.title("👤 Employee Risk Lookup")
    search = st.text_input("Search by Employee ID, Name, or Department")
    df_show = df_scored[["EmployeeID","FirstName","LastName","Department","Level","Location","Gender",
                          "YearsAtCompany","MarketRateRatio","OverTime","AttritionProb","RiskCategory"]].copy()
    df_show["AttritionProb"] = df_show["AttritionProb"].apply(lambda x:f"{x:.1%}")

    if search:
        mask = (df_scored["EmployeeID"].str.contains(search, case=False) |
                df_scored["FirstName"].str.contains(search, case=False) |
                df_scored["LastName"].str.contains(search, case=False) |
                df_scored["Department"].str.contains(search, case=False))
        df_show = df_show[mask]

    risk_filter = st.selectbox("Filter by Risk", ["All","Critical","High","Medium","Low"])
    if risk_filter!="All":
        df_show = df_show[df_show["RiskCategory"]==risk_filter]

    st.dataframe(df_show.sort_values("AttritionProb",ascending=False).head(100),
                 use_container_width=True, height=500)
    st.caption(f"Showing {len(df_show):,} employees")

st.markdown("---")
st.caption("TechNova Inc. HR Analytics · Powered by Machine Learning · Built with Streamlit")
