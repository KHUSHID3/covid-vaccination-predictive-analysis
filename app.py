import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="COVID Vaccination Predictive Analysis",
    layout="wide"
)

# ---------------- Title ----------------
st.title("COVID-19 Vaccination Predictive Analysis")
st.write("EDA, Machine Learning, PCA, and Clustering based analysis")

# ---------------- Load Dataset ----------------
df = pd.read_csv("COVID-19_Vaccinations_by_Age_and_Race-Ethnicity_-_Historical.csv")

# ---------------- Dataset Preview ----------------
st.header("Dataset Preview")
st.dataframe(df.head())

# ================== EDA SECTION ==================
st.header("Exploratory Data Analysis (EDA)")

# ---- Histogram (Distribution) ----
st.subheader("Distribution of 1st Dose Vaccination")

fig1, ax1 = plt.subplots()
sns.histplot(df["1st Dose"].dropna(), bins=30, kde=True, ax=ax1)
ax1.set_xlabel("1st Dose Count")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

# ---- Correlation Heatmap ----
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=["float64", "int64"])

fig2, ax2 = plt.subplots(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False,
    ax=ax2
)
st.pyplot(fig2)

# ================== PROJECT HIGHLIGHTS ==================
st.header("Project Highlights")
st.markdown("""
- Exploratory Data Analysis (EDA)
- Supervised Learning: Linear Regression & XGBoost
- Unsupervised Learning: K-Means & Hierarchical Clustering
- PCA for Dimensionality Reduction
- High accuracy predictive models
""")

st.success("EDA section added successfully 🎉")
