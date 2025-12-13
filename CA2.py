#IMPORT LIBARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#LOAD DATASET
df = pd.read_csv("COVID-19_Vaccinations_by_Age_and_Race-Ethnicity_-_Historical.csv")
print(df.head(10))
print(df.info())
print(df.describe())
#HANDLE MISSING VALUES
print(df.isna().sum())
df = df.dropna(subset=["Week End"])
df = df.fillna(0)
print(df.isna().sum())
print("Unique Age Groups:")
print(df["Age Group"].unique())
print("\nUnique Race/Ethnicity Groups:")
print(df["Race/Ethnicity"].unique())
print("\nDescriptive summary (numeric):")
print(df.describe())
#EDA
#STEP 1 — Vaccination Trend Over Time (Line Chart)
df = df.sort_values("Week End")
# convert date
df["Week End"] = pd.to_datetime(df["Week End"])
# create month column
df["Month"] = df["Week End"].dt.to_period("M")
# monthly 1st dose sum
monthly = df.groupby("Month")["1st Dose"].sum()
plt.figure(figsize=(10,4))
plt.plot(monthly.index.astype(str), monthly.values, marker="o")
plt.title("Monthly Trend of 1st Dose Vaccinations")
plt.xlabel("Month")
plt.ylabel("Total 1st Dose")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#STEP 2 — Age Group Comparison (Bar Chart)
age_data = df.groupby("Age Group")["1st Dose"].sum().sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=age_data.values, y=age_data.index)
plt.title("Vaccination Comparison Across Age Groups")
plt.xlabel("1st Dose Total")
plt.ylabel("Age Group")
plt.tight_layout()
plt.show()
#STEP 3 — Race/Ethnicity Comparison (Bar Chart)
race_data = df.groupby("Race/Ethnicity")["1st Dose"].sum().sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=race_data.values, y=race_data.index)
plt.title("Vaccination Comparison Across Race/Ethnicity Groups")
plt.xlabel("1st Dose Total")
plt.ylabel("Race/Ethnicity")
plt.tight_layout()
plt.show()
# STEP: Outlier Detection for 1st Dose
#Basic Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=df["1st Dose"])
plt.title("Boxplot - 1st Dose")
plt.xlabel("1st Dose Count")
plt.tight_layout()
plt.show()
#IQR Method for detecting exact outlier values
Q1 = df["1st Dose"].quantile(0.25)
Q3 = df["1st Dose"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
outliers = df[(df["1st Dose"] < lower_limit) | (df["1st Dose"] > upper_limit)]
print("Total outliers detected:", len(outliers))
print("\nSample outliers:")
print(outliers[["Week End", "Age Group", "Race/Ethnicity", "1st Dose"]].head())
#Outlier Scatter Plot
plt.figure(figsize=(10,4))
normal = df[(df["1st Dose"] >= lower_limit) & (df["1st Dose"] <= upper_limit)]
plt.scatter(normal.index, normal["1st Dose"], color="blue", s=10, label="Normal Data")
# outlier points
plt.scatter(outliers.index, outliers["1st Dose"], color="red", s=20, label="Outliers")
plt.title("Outlier Detection in 1st Dose (Scatter Plot)")
plt.xlabel("Index")
plt.ylabel("1st Dose Count")
plt.legend()
plt.tight_layout()
plt.show()
#Outliers Remove
df_clean = df[(df["1st Dose"] >= lower_limit) & (df["1st Dose"] <= upper_limit)]
print("New dataset shape after removing outliers:", df_clean.shape)
plt.figure(figsize=(10,4))
plt.scatter(df_clean.index, df_clean["1st Dose"], s=10)
plt.title("1st Dose After Removing Outliers")
plt.xlabel("Index")
plt.ylabel("1st Dose Count")
plt.tight_layout()
plt.show()
# Histogram for 1st Dose
plt.figure(figsize=(6,4))
sns.histplot(df["1st Dose"], bins=30, kde=True, edgecolor='black')
plt.title("Histogram with KDE Curve for 1st Dose")
plt.xlabel("1st Dose Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
#Correlation Matrix
numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])
print(numeric_df.head())
corr = numeric_df.corr()
print(corr)
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()
#VIF
num_df = df_clean.select_dtypes(include=['float64', 'int64'])
# target column remove
num_df = num_df.drop(columns=["1st Dose"], errors='ignore')
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
vif = pd.DataFrame()
vif["Feature"] = num_df.columns
vif["VIF"] = [variance_inflation_factor(num_df.values, i)
              for i in range(num_df.shape[1])]
print(vif)
#MODELLING
#Simple Feature Scaling
df_clean = df_clean.copy()
# Convert Week End to datetime
df_clean.loc[:, "Week End"] = pd.to_datetime(df_clean["Week End"], errors="coerce")
# Create numeric timestamp
df_clean.loc[:, "Week_End_Num"] = df_clean["Week End"].astype("int64") // 10**9
# One-hot encoding
df_encoded = pd.get_dummies(df_clean,columns=["Age Group", "Race/Ethnicity"],drop_first=True)
# Replace X, y with numeric-only version, then split again
X = df_encoded.select_dtypes(include=[np.number]).drop(columns=["1st Dose"])
y = df_encoded["1st Dose"]
#TRAIN_TEST_SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
#SUPERVISED LEARNING
#1ST-LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
#2ND - XG BOOST
from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    eval_metric="rmse"   # Only this is needed
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
#EVALUATION OF LINEAR REGRESSION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("\nLinear Regression Evaluation:")
print("MAE :", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R²  :", r2_score(y_test, y_pred_lr))
#EVALUATION OF XG BOOST
print("\nXGBoost Evaluation:")
print("MAE :", mean_absolute_error(y_test, y_pred_xgb))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("R²  :", r2_score(y_test, y_pred_xgb))
#Feature Importance plot (XGBoost)
from xgboost import plot_importance
plt.figure(figsize=(10,6))
plot_importance(xgb, max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.show()
#Graph of Actual vs Predicted
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.xlabel("Actual 1st Dose Count")
plt.ylabel("Predicted 1st Dose Count")
plt.title("Actual vs Predicted (XGBoost)")
plt.show()
#UNSUPERVISED MODEL
# 1ST — K-MEANS CLUSTERING
X_unsup = df_encoded.select_dtypes(include=['number'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled_unsup = scaler.fit_transform(X_unsup)
#ELBOW METHOD(TO FIND OPTIMAL K)
from sklearn.cluster import KMeans
inertia = []
K = range(1,8)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled_unsup)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(7,4))
plt.plot(K, inertia, marker='o')
plt.xlabel("No. of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()
#Fit K-Means (Assume K = 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled_unsup)
df_encoded['Cluster_KMeans'] = clusters
print(df_encoded.groupby('Cluster_KMeans').mean())
#UNSUPERVISED MODEL
#2ND — HIERARCHICAL CLUSTERING
X_scaled_unsup = scaler.fit_transform(X_unsup)
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(12,6))
sch.dendrogram(sch.linkage(X_scaled_unsup, method='ward'),
               truncate_mode='lastp',  # show only last p merges
               p=30,                   # try 20-50 for readability
               leaf_rotation=90.,
               leaf_font_size=10.)
plt.title("Truncated dendrogram (last 30 merges)")
plt.xlabel("Cluster index or (sample index)")
plt.ylabel("Distance")
plt.show()
#AGGLOMERATIVE CLUSTERING
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters_hc = hc.fit_predict(X_scaled_unsup)
df_encoded["Cluster_Hierarchical"] = clusters_hc
print(df_encoded.groupby("Cluster_Hierarchical").mean())
#SUMMARY OF CLUSTER AND CLUSTERS
cluster_summary = df_encoded.groupby('Cluster_KMeans').mean().T
print(cluster_summary)
print(df_encoded.groupby('Cluster_KMeans')[['1st Dose','Vaccine Series Completed','Boosted']].mean())
#PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.show()
#2D scatter
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
plt.figure(figsize=(7,5))
plt.scatter(X_pca2[:,0], X_pca2[:,1], s=10, alpha=0.6)
plt.title("PCA - First Two Components")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()












