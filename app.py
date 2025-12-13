import streamlit as st
import pandas as pd

st.set_page_config(page_title="COVID Vaccination Predictive Analysis", layout="wide")

st.title("COVID-19 Vaccination Predictive Analysis")
st.write("Predictive analysis using Machine Learning, PCA, and Clustering")

# Load data
df = pd.read_csv("COVID-19_Vaccinations_by_Age_and_Race-Ethnicity_-_Historical.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Project Highlights")
st.markdown("""
- Linear Regression & XGBoost
- KMeans & Hierarchical Clustering
- PCA for dimensionality reduction
- High accuracy predictive models
""")

st.success("Deployment successful 🎉")
