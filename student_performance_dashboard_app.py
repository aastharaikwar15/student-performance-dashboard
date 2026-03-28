import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Student Performance Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Custom CSS for animated UI
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #f8fafc !important;
        color: #0f172a !important;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 50%, #ecfeff 100%) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e0e7ff 0%, #f8fafc 100%) !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2.2s ease-in-out infinite alternate;
    }

    .sub-text {
        font-size: 16px;
        color: #334155 !important;
        margin-bottom: 18px;
    }

    .metric-box {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 18px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        color: #0f172a !important;
    }

    .metric-box h3, .metric-box p {
        color: #0f172a !important;
        margin: 0;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #1e293b !important;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        color: #0f172a !important;
    }

    label, .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #0f172a !important;
        font-weight: 600;
    }

    .stMarkdown, .stText, p, div {
        color: inherit;
    }

    @keyframes glow {
        from { text-shadow: 0 0 8px rgba(59,130,246,0.18); }
        to { text-shadow: 0 0 18px rgba(168,85,247,0.35); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data generation
# -----------------------------
@st.cache_data

def generate_student_data(n=100, seed=42):
    np.random.seed(seed)

    student_ids = [f"STU{str(i).zfill(3)}" for i in range(1, n + 1)]
    gender = np.random.choice(["Male", "Female"], size=n)
    course = np.random.choice(
        ["BTech CSE", "BTech DS", "BCA", "BSc Data Science", "BBA"],
        size=n,
        p=[0.28, 0.22, 0.18, 0.17, 0.15],
    )
    year = np.random.choice([1, 2, 3, 4], size=n, p=[0.28, 0.27, 0.24, 0.21])

    previous_score = np.clip(np.random.normal(72, 12, n), 40, 98)
    attendance = np.clip(np.random.normal(80, 11, n), 45, 100)
    study_hours = np.clip(np.random.normal(5.5, 2.3, n), 1, 12)
    assignment_rate = np.clip(np.random.normal(78, 15, n), 35, 100)
    internal_score = np.clip(np.random.normal(74, 13, n), 38, 100)
    sleep_hours = np.clip(np.random.normal(6.9, 1.1, n), 4, 9)
    screen_time = np.clip(np.random.normal(5.2, 1.9, n), 1.5, 10)
    extracurricular = np.clip(np.random.normal(2.8, 1.7, n), 0, 8)

    noise = np.random.normal(0, 4.5, n)

    final_score = (
        0.50 * previous_score
        + 0.14 * attendance
        + 0.10 * assignment_rate
        + 0.12 * internal_score
        + 0.08 * (study_hours * 10)
        + 0.03 * (sleep_hours * 10)
        - 0.03 * (screen_time * 10)
        + 0.02 * (extracurricular * 10)
        + noise
    )
    final_score = np.clip(final_score, 35, 99)

    need_attention = np.where(
        (final_score < 60) | (attendance < 65) | (assignment_rate < 55),
        "Yes",
        "No",
    )

    performance_band = pd.cut(
        final_score,
        bins=[0, 50, 65, 80, 100],
        labels=["At Risk", "Average", "Good", "Excellent"],
    )

    df = pd.DataFrame(
        {
            "student_id": student_ids,
            "gender": gender,
            "course": course,
            "year": year,
            "previous_score": np.round(previous_score, 1),
            "attendance": np.round(attendance, 1),
            "study_hours": np.round(study_hours, 1),
            "assignment_rate": np.round(assignment_rate, 1),
            "internal_score": np.round(internal_score, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "screen_time": np.round(screen_time, 1),
            "extracurricular_hours": np.round(extracurricular, 1),
            "final_score": np.round(final_score, 1),
            "need_attention": need_attention,
            "performance_band": performance_band.astype(str),
        }
    )

    # Add a few missing values intentionally for preprocessing demo
    missing_rows = np.random.choice(df.index, 8, replace=False)
    df.loc[missing_rows[:3], "attendance"] = np.nan
    df.loc[missing_rows[3:5], "study_hours"] = np.nan
    df.loc[missing_rows[5:7], "assignment_rate"] = np.nan
    df.loc[missing_rows[7:], "internal_score"] = np.nan

    return df


# -----------------------------
# Preprocessing and modeling
# -----------------------------
@st.cache_data

def run_analysis(df):
    working_df = df.copy()

    numeric_cols = [
        "year",
        "previous_score",
        "attendance",
        "study_hours",
        "assignment_rate",
        "internal_score",
        "sleep_hours",
        "screen_time",
        "extracurricular_hours",
    ]
    categorical_cols = ["gender", "course"]

    # Regression target
    X_reg = working_df[numeric_cols + categorical_cols]
    y_reg = working_df["final_score"]

    # Classification target
    y_cls = (working_df["need_attention"] == "Yes").astype(int)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_reg, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # One-hot after preprocessor manually through get_dummies on imputed full dataset for easier feature names
    filled = working_df.copy()
    for col in ["attendance", "study_hours", "assignment_rate", "internal_score"]:
        filled[col] = filled[col].fillna(filled[col].median())

    X_full = pd.get_dummies(filled[numeric_cols + categorical_cols], drop_first=True)
    y_full_reg = filled["final_score"]
    y_full_cls = (filled["need_attention"] == "Yes").astype(int)

    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_full, y_full_reg, test_size=0.20, random_state=42)
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_full, y_full_cls, test_size=0.20, random_state=42, stratify=y_full_cls)

    reg_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg_model.fit(Xtr_r, ytr_r)
    reg_pred = reg_model.predict(Xte_r)

    rf_model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    rf_model.fit(Xtr_r, ytr_r)
    rf_pred = rf_model.predict(Xte_r)

    cls_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    cls_model.fit(Xtr_c, ytr_c)
    cls_pred = cls_model.predict(Xte_c)

    clustering_features = filled[
        [
            "previous_score",
            "attendance",
            "study_hours",
            "assignment_rate",
            "internal_score",
            "screen_time",
        ]
    ]
    clustering_scaled = StandardScaler().fit_transform(clustering_features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(clustering_scaled)
    silhouette = silhouette_score(clustering_scaled, clusters)
    filled["cluster"] = clusters

    cluster_map = {
        0: "Balanced Learners",
        1: "High Achievers",
        2: "Needs Support",
    }
    filled["student_segment"] = filled["cluster"].map(cluster_map)

    reg_metrics = {
        "Decision Tree MAE": mean_absolute_error(yte_r, reg_pred),
        "Decision Tree RMSE": np.sqrt(mean_squared_error(yte_r, reg_pred)),
        "Decision Tree R2": r2_score(yte_r, reg_pred),
        "Random Forest MAE": mean_absolute_error(yte_r, rf_pred),
        "Random Forest RMSE": np.sqrt(mean_squared_error(yte_r, rf_pred)),
        "Random Forest R2": r2_score(yte_r, rf_pred),
    }

    cls_metrics = {
        "Accuracy": accuracy_score(yte_c, cls_pred),
        "Precision": precision_score(yte_c, cls_pred, zero_division=0),
        "Recall": recall_score(yte_c, cls_pred, zero_division=0),
        "F1 Score": f1_score(yte_c, cls_pred, zero_division=0),
    }

    cm = confusion_matrix(yte_c, cls_pred)

    feature_importance = pd.DataFrame(
        {
            "feature": X_full.columns,
            "importance": reg_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return {
        "clean_df": filled,
        "reg_model": reg_model,
        "cls_model": cls_model,
        "reg_metrics": reg_metrics,
        "cls_metrics": cls_metrics,
        "conf_matrix": cm,
        "feature_importance": feature_importance,
        "silhouette": silhouette,
        "X_columns": X_full.columns.tolist(),
    }


# -----------------------------
# Main App
# -----------------------------
df = generate_student_data(100)
results = run_analysis(df)
clean_df = results["clean_df"]

st.markdown('<div class="main-title">🎓 Student Performance Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Final semester Data Science project using only Python, Data Mining concepts, Decision Tree Regression, Classification, Clustering, preprocessing, and interactive dashboard visualizations.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    course_filter = st.multiselect(
        "Select course",
        options=sorted(clean_df["course"].unique()),
        default=sorted(clean_df["course"].unique()),
    )
    year_filter = st.multiselect(
        "Select year",
        options=sorted(clean_df["year"].unique()),
        default=sorted(clean_df["year"].unique()),
    )
    show_raw = st.checkbox("Show raw dataset", value=False)

filtered_df = clean_df[
    (clean_df["course"].isin(course_filter)) & (clean_df["year"].isin(year_filter))
]

avg_final = filtered_df["final_score"].mean()
avg_attendance = filtered_df["attendance"].mean()
attention_count = (filtered_df["need_attention"] == "Yes").sum()
top_students = (filtered_df["performance_band"] == "Excellent").sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-box"><h3>{len(filtered_df)}</h3><p>Total Students</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box"><h3>{avg_final:.1f}</h3><p>Average Final Score</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box"><h3>{avg_attendance:.1f}%</h3><p>Average Attendance</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-box"><h3>{attention_count}</h3><p>Need Attention</p></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">📊 Data Preview</div>', unsafe_allow_html=True)
if show_raw:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(filtered_df.head(10), use_container_width=True)

left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">🎯 Final Score Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        filtered_df,
        x="final_score",
        nbins=18,
        marginal="box",
        title="Distribution of Final Scores",
    )
    fig_hist.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_hist, use_container_width=True)

with right:
    st.markdown('<div class="section-title">📌 Performance Band Count</div>', unsafe_allow_html=True)
    band_count = filtered_df["performance_band"].value_counts().reset_index()
    band_count.columns = ["performance_band", "count"]
    fig_pie = px.pie(band_count, names="performance_band", values="count", hole=0.45, title="Performance Categories")
    fig_pie.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_pie, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = [
        "previous_score",
        "attendance",
        "study_hours",
        "assignment_rate",
        "internal_score",
        "sleep_hours",
        "screen_time",
        "extracurricular_hours",
        "final_score",
    ]
    corr = filtered_df[corr_cols].corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
    fig_corr.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

with row2_col2:
    st.markdown('<div class="section-title">🌟 Feature Importance (Decision Tree)</div>', unsafe_allow_html=True)
    fi = results["feature_importance"].head(10)
    fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", title="Top Features Affecting Final Score")
    fig_fi.update_layout(template="plotly_white", height=500, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)
mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.metric("Decision Tree R²", f"{results['reg_metrics']['Decision Tree R2']:.3f}")
    st.metric("Decision Tree RMSE", f"{results['reg_metrics']['Decision Tree RMSE']:.2f}")
with mc2:
    st.metric("Random Forest R²", f"{results['reg_metrics']['Random Forest R2']:.3f}")
    st.metric("Random Forest RMSE", f"{results['reg_metrics']['Random Forest RMSE']:.2f}")
with mc3:
    st.metric("Attention Classifier Accuracy", f"{results['cls_metrics']['Accuracy']:.3f}")
    st.metric("Attention Classifier F1", f"{results['cls_metrics']['F1 Score']:.3f}")

cm = results["conf_matrix"]
cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Pred No", "Pred Yes"])
fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix - Need Attention Prediction")
fig_cm.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig_cm, use_container_width=True)

st.markdown('<div class="section-title">🧠 Student Segmentation (Clustering)</div>', unsafe_allow_html=True)
seg_count = filtered_df["student_segment"].value_counts().reset_index()
seg_count.columns = ["segment", "count"]
seg_fig = px.bar(seg_count, x="segment", y="count", title=f"KMeans Student Segments | Silhouette Score: {results['silhouette']:.3f}")
seg_fig.update_layout(template="plotly_white", height=420)
st.plotly_chart(seg_fig, use_container_width=True)

st.markdown('<div class="section-title">🔮 Predict New Student Final Score</div>', unsafe_allow_html=True)
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        in_previous = st.slider("Previous Score", 40.0, 100.0, 72.0)
        in_attendance = st.slider("Attendance %", 40.0, 100.0, 80.0)
        in_study = st.slider("Study Hours", 1.0, 12.0, 5.0)
    with col2:
        in_assign = st.slider("Assignment Completion %", 30.0, 100.0, 75.0)
        in_internal = st.slider("Internal Score", 30.0, 100.0, 74.0)
        in_sleep = st.slider("Sleep Hours", 4.0, 9.0, 7.0)
    with col3:
        in_screen = st.slider("Screen Time", 1.0, 10.0, 5.0)
        in_extra = st.slider("Extracurricular Hours", 0.0, 8.0, 3.0)
        in_year = st.selectbox("Year", [1, 2, 3, 4])
    in_gender = st.selectbox("Gender", ["Male", "Female"])
    in_course = st.selectbox("Course", ["BTech CSE", "BTech DS", "BCA", "BSc Data Science", "BBA"])

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    new_df = pd.DataFrame(
        {
            "year": [in_year],
            "previous_score": [in_previous],
            "attendance": [in_attendance],
            "study_hours": [in_study],
            "assignment_rate": [in_assign],
            "internal_score": [in_internal],
            "sleep_hours": [in_sleep],
            "screen_time": [in_screen],
            "extracurricular_hours": [in_extra],
            "gender": [in_gender],
            "course": [in_course],
        }
    )

    train_ready = pd.get_dummies(clean_df[
        [
            "year",
            "previous_score",
            "attendance",
            "study_hours",
            "assignment_rate",
            "internal_score",
            "sleep_hours",
            "screen_time",
            "extracurricular_hours",
            "gender",
            "course",
        ]
    ], drop_first=True)

    model_train_y = clean_df["final_score"]
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(train_ready, model_train_y)

    new_ready = pd.get_dummies(new_df, drop_first=True)
    new_ready = new_ready.reindex(columns=train_ready.columns, fill_value=0)
    pred_score = model.predict(new_ready)[0]

    attention_flag = "Yes" if pred_score < 60 or in_attendance < 65 or in_assign < 55 else "No"
    band = (
        "At Risk" if pred_score < 50 else
        "Average" if pred_score < 65 else
        "Good" if pred_score < 80 else
        "Excellent"
    )

    st.success(f"Predicted Final Score: {pred_score:.2f}")
    st.info(f"Need Attention: {attention_flag}")
    st.warning(f"Performance Band: {band}")

st.markdown('<div class="section-title">📘 Data Mining Concepts Used</div>', unsafe_allow_html=True)
with st.expander("Click to view project concepts"):
    st.write("""
    - Data Generation / Dataset Creation for 100 students
    - Data Cleaning with missing value handling
    - Feature Engineering using academic and behavioral factors
    - Exploratory Data Analysis (distribution, correlations, segments)
    - Decision Tree Regression for final score prediction
    - Decision Tree Classification for need-attention prediction
    - Random Forest Regression for model comparison
    - KMeans Clustering for student segmentation
    - Feature Importance analysis
    - Interactive dashboard for insight generation
    """)

st.caption("Made fully in Python for VS Code using Streamlit, Pandas, Scikit-learn, and Plotly.")
