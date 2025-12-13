# ===============================
# COVID-19 Vaccination Predictive Analysis
# Full EDA + PCA + Regression + Clustering
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="COVID Vaccination Predictive Analysis",
    layout="wide"
)

st.title("COVID-19 Vaccination Predictive Analysis")
st.write("EDA, Regression, PCA & Clustering using Machine Learning")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "COVID-19_Vaccinations_by_Age_and_Race-Ethnicity_-_Historical.csv"
    )
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# DATA CLEANING
# -------------------------------
df["Week End"] = pd.to_datetime(df["Week End"], errors="coerce")
df = df.dropna(subset=["Week End"])
df = df.fillna(0)

# -------------------------------
# EDA SECTION
# -------------------------------
st.header("📊 Exploratory Data Analysis (EDA)")

# ---- Monthly Trend
st.subheader("Monthly Trend of 1st Dose Vaccination")
df_sorted = df.sort_values("Week End")
df_sorted["Month"] = df_sorted["Week End"].dt.to_period("M")
monthly = df_sorted.groupby("Month")["1st Dose"].sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(monthly.index.astype(str), monthly.values, marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Total 1st Dose")
ax.set_title("Monthly Trend of 1st Dose Vaccinations")
plt.xticks(rotation=45)
st.pyplot(fig)

# ---- Age Group Bar Plot
st.subheader("Vaccination by Age Group")
age_data = df.groupby("Age Group")["1st Dose"].sum().sort_values()

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=age_data.values, y=age_data.index, ax=ax)
ax.set_title("1st Dose by Age Group")
st.pyplot(fig)

# ---- Race/Ethnicity Bar Plot
st.subheader("Vaccination by Race / Ethnicity")
race_data = df.groupby("Race/Ethnicity")["1st Dose"].sum().sort_values()

fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=race_data.values, y=race_data.index, ax=ax)
ax.set_title("1st Dose by Race/Ethnicity")
st.pyplot(fig)

# -------------------------------
# OUTLIER ANALYSIS
# -------------------------------
st.header("🚨 Outlier Analysis")

Q1 = df["1st Dose"].quantile(0.25)
Q3 = df["1st Dose"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["1st Dose"] < lower) | (df["1st Dose"] > upper)]
df_clean = df[(df["1st Dose"] >= lower) & (df["1st Dose"] <= upper)]

st.write(f"Total outliers detected: **{len(outliers)}**")

# ---- Scatter with outliers
fig, ax = plt.subplots(figsize=(10,4))
ax.scatter(df_clean.index, df_clean["1st Dose"], s=10, label="Normal", alpha=0.6)
ax.scatter(outliers.index, outliers["1st Dose"], color="red", s=15, label="Outliers")
ax.set_title("Outlier Detection (Scatter)")
ax.legend()
st.pyplot(fig)

# ---- Scatter after removing outliers
fig, ax = plt.subplots(figsize=(10,4))
ax.scatter(df_clean.index, df_clean["1st Dose"], s=10)
ax.set_title("1st Dose after Removing Outliers")
st.pyplot(fig)

# ---- Histogram
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(df_clean["1st Dose"], bins=30, kde=True, ax=ax)
ax.set_title("Histogram of 1st Dose (Cleaned)")
st.pyplot(fig)

# -------------------------------
# CORRELATION HEATMAP
# -------------------------------
st.header("📌 Correlation Heatmap")

numeric_df = df_clean.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df_clean["Week_End_Num"] = df_clean["Week End"].astype("int64") // 10**9
df_encoded = pd.get_dummies(
    df_clean,
    columns=["Age Group", "Race/Ethnicity"],
    drop_first=True
)

X = df_encoded.select_dtypes(include=[np.number]).drop(columns=["1st Dose"])
y = df_encoded["1st Dose"]

# -------------------------------
# REGRESSION MODELS
# -------------------------------
st.header("📈 Regression Models")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ---- XGBoost
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    eval_metric="rmse"
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# ---- Metrics
st.subheader("Model Evaluation")

st.write("### Linear Regression")
st.write("MAE:", mean_absolute_error(y_test, y_pred_lr))
st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
st.write("R²:", r2_score(y_test, y_pred_lr))

st.write("### XGBoost")
st.write("MAE:", mean_absolute_error(y_test, y_pred_xgb))
st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
st.write("R²:", r2_score(y_test, y_pred_xgb))

# ---- Actual vs Predicted
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(y_test, y_pred_xgb, alpha=0.5)
ax.set_xlabel("Actual 1st Dose")
ax.set_ylabel("Predicted 1st Dose")
ax.set_title("Actual vs Predicted (XGBoost)")
st.pyplot(fig)

# -------------------------------
# PCA
# -------------------------------
st.header("🔻 Principal Component Analysis (PCA)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
    marker="o"
)
ax.set_title("Cumulative Explained Variance")
ax.set_xlabel("Components")
ax.set_ylabel("Variance")
st.pyplot(fig)

# ---- PCA 2D Scatter
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(X_pca2[:,0], X_pca2[:,1], s=10, alpha=0.6)
ax.set_title("PCA – First Two Components")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)

# -------------------------------
# CLUSTERING
# -------------------------------
st.header("🧩 KMeans Clustering")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(
    X_pca2[:,0],
    X_pca2[:,1],
    c=clusters,
    cmap="viridis",
    s=10
)
ax.set_title("KMeans Clusters (PCA space)")
st.pyplot(fig)

st.success("🎉 Full EDA, PCA, Regression & Clustering Analysis Completed!")
