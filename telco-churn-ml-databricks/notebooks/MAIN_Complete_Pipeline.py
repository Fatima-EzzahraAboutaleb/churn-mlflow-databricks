# Databricks notebook source
# MAGIC %md
# MAGIC # Telco Customer Churn Prediction
# MAGIC
# MAGIC **Author:** Fatima-Ezzahra ABOUTALEB  
# MAGIC **Course:** Machine Learning  
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Setup & Imports

# COMMAND ----------

# MAGIC %pip install -r /Workspace/Users/fatimaezzahraaboutaleb5@gmail.com/telco-churn-ml-databricks/requirements.txt
# MAGIC

# COMMAND ----------

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("✓ All libraries imported successfully!")
print(f"✓ MLflow version: {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##  1. Load Data

# COMMAND ----------

# Load dataset from DBFS or local path
# For Databricks, upload the CSV to DBFS first
# DATA_PATH = "/dbfs/FileStore/tables/WA_Fn_UseC__Telco_Customer_Churn.csv"

# Alternative paths:
# DATA_PATH = "dbfs:/FileStore/tables/WA_Fn_UseC__Telco_Customer_Churn.csv"
DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"  # For local

df = pd.read_csv(DATA_PATH)

print("="*80)
print("DATASET LOADED")
print("="*80)
print(f"Shape: {df.shape}")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print("\nFirst 5 rows:")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📈 Dataset Overview

# COMMAND ----------

# Basic information
print("="*80)
print("DATASET INFORMATION")
print("="*80)
print(df.info())

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
print(df.isnull().sum())

print("\n" + "="*80)
print("CHURN DISTRIBUTION")
print("="*80)
print(df['Churn'].value_counts())
print(f"\nChurn Rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Cleaning & Preprocessing

# COMMAND ----------

# Create a copy
df_clean = df.copy()

# 1. Remove customerID
if 'customerID' in df_clean.columns:
    df_clean.drop('customerID', axis=1, inplace=True)
    print("✓ Removed customerID column")

# 2. Convert TotalCharges to numeric
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
print(f"✓ Converted TotalCharges to numeric")

# 3. Handle missing values
missing_count = df_clean['TotalCharges'].isnull().sum()
if missing_count > 0:
    print(f"⚠ Found {missing_count} missing values in TotalCharges")
    df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
    print(f"✓ Filled missing values with median")

# 4. Identify feature types
numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Churn' in numerical_features:
    numerical_features.remove('Churn')

categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
if 'Churn' in categorical_features:
    categorical_features.remove('Churn')

print(f"\n✓ Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"✓ Categorical features ({len(categorical_features)}): {categorical_features[:5]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Exploratory Data Analysis (EDA)

# COMMAND ----------

# Visualize Churn distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
churn_counts = df_clean['Churn'].value_counts()
axes[0].bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'], alpha=0.8)
axes[0].set_xlabel('Churn', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontsize=12, fontweight='bold')

# Pie chart
axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Churn Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# Numerical features distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, feature in enumerate(numerical_features):
    axes[idx].hist(df_clean[feature], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_xlabel(feature, fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Convert Churn to numeric for correlation
df_corr = df_clean.copy()
df_corr['Churn'] = (df_corr['Churn'] == 'Yes').astype(int)

# Calculate correlation for numerical features
corr_matrix = df_corr[numerical_features + ['Churn']].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering & Encoding

# COMMAND ----------

# Encode categorical features
df_encoded = df_clean.copy()

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_features)} categorical features")

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df_encoded['Churn'])
X = df_encoded.drop('Churn', axis=1)

print(f"✓ Target encoded: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
print(f"\nFinal dataset shape: {X.shape}")
print(f"Features: {X.columns.tolist()[:5]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train-Test Split

# COMMAND ----------

# Split data (80-20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*80)
print("TRAIN-TEST SPLIT")
print("="*80)
print(f"Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nClass distribution in train set:")
print(f"  No Churn: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
print(f"  Churn:    {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ##  5. Feature Scaling

# COMMAND ----------

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("✓ Features scaled using StandardScaler")
print(f"  Training set: {X_train_scaled.shape}")
print(f"  Test set: {X_test_scaled.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC #  PART 2: Dimensionality Reduction

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. PCA (Principal Component Analysis)

# COMMAND ----------

from sklearn.decomposition import PCA

# Fit PCA with all components to see explained variance
pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Variance per component
explained_var = pca_full.explained_variance_ratio_
axes[0].bar(range(1, len(explained_var) + 1), explained_var, color='steelblue', alpha=0.8)
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[0].set_title('PCA - Explained Variance per Component', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Cumulative variance
cumulative_var = np.cumsum(explained_var)
axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o', linewidth=2, markersize=8)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance', linewidth=2)
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[1].set_title('PCA - Cumulative Explained Variance', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\n✓ {n_components_95} components explain 95% of the variance")

# COMMAND ----------

# Apply PCA with 2 components for visualization
pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_train_scaled)

print(f"✓ PCA 2D applied")
print(f"  Explained variance: {pca_2d.explained_variance_ratio_}")
print(f"  Total variance explained: {pca_2d.explained_variance_ratio_.sum():.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title('PCA - 2D Projection (Colored by Churn)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_label('Churn', fontsize=11)
cbar.ax.set_yticklabels(['No', 'Yes'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##  7. t-SNE (t-Distributed Stochastic Neighbor Embedding)

# COMMAND ----------

from sklearn.manifold import TSNE

# Apply t-SNE (subsample for speed if dataset is large)
sample_size = min(3000, len(X_train_scaled))
indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
X_train_sample = X_train_scaled.iloc[indices].values
y_train_sample = y_train[indices]

print(f"Applying t-SNE on {sample_size} samples...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=0)
X_tsne = tsne.fit_transform(X_train_sample)

print(f"✓ t-SNE complete!")
print(f"  KL divergence: {tsne.kl_divergence_:.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train_sample, cmap='coolwarm', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
ax.set_xlabel('t-SNE Component 1', fontsize=12)
ax.set_ylabel('t-SNE Component 2', fontsize=12)
ax.set_title('t-SNE - 2D Projection (Colored by Churn)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_label('Churn', fontsize=11)
cbar.ax.set_yticklabels(['No', 'Yes'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##  8. NMF (Non-negative Matrix Factorization)

# COMMAND ----------

from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

# NMF requires non-negative data
minmax_scaler = MinMaxScaler()
X_train_nonneg = minmax_scaler.fit_transform(X_train)

# Apply NMF
nmf = NMF(n_components=2, random_state=42, max_iter=500)
X_nmf = nmf.fit_transform(X_train_nonneg)

print(f"✓ NMF complete!")
print(f"  Reconstruction error: {nmf.reconstruction_err_:.4f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_nmf[:, 0], X_nmf[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
ax.set_xlabel('NMF Component 1', fontsize=12)
ax.set_ylabel('NMF Component 2', fontsize=12)
ax.set_title('NMF - 2D Projection (Colored by Churn)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, ticks=[0, 1])
cbar.set_label('Churn', fontsize=11)
cbar.ax.set_yticklabels(['No', 'Yes'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #  PART 3: Unsupervised Learning (Clustering)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  9. K-Means Clustering

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Find optimal k using Elbow method
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_train_scaled, kmeans.labels_))

# Plot Elbow & Silhouette
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Method for Optimal k', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score for Optimal k', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Mark best k
best_k = k_range[np.argmax(silhouette_scores)]
axes[1].scatter([best_k], [max(silhouette_scores)], color='red', s=200, zorder=5, marker='*', label=f'Best k={best_k}')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"✓ Optimal k: {best_k} (Silhouette Score: {max(silhouette_scores):.4f})")

# COMMAND ----------

# Fit K-Means with optimal k
optimal_k = 3  # Or use best_k from above
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_train_scaled)

# Evaluation
sil_score = silhouette_score(X_train_scaled, kmeans_labels)
db_score = davies_bouldin_score(X_train_scaled, kmeans_labels)
ch_score = calinski_harabasz_score(X_train_scaled, kmeans_labels)

print("="*80)
print(f"K-MEANS CLUSTERING (k={optimal_k})")
print("="*80)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")

# Compare with true labels
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(y_train, kmeans_labels)
nmi = normalized_mutual_info_score(y_train, kmeans_labels)
print(f"\nComparison with true labels:")
print(f"  Adjusted Rand Index: {ari:.4f}")
print(f"  Normalized Mutual Info: {nmi:.4f}")

# COMMAND ----------

# Visualize clusters on PCA
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title(f'K-Means Clustering (k={optimal_k}) - PCA Projection', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##  10. DBSCAN (Density-Based Clustering)

# COMMAND ----------

from sklearn.cluster import DBSCAN

# Fit DBSCAN
dbscan = DBSCAN(eps=2.0, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_train_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print("="*80)
print("DBSCAN CLUSTERING")
print("="*80)
print(f"Number of clusters: {n_clusters_dbscan}")
print(f"Noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.2f}%)")

if n_clusters_dbscan > 1:
    # Calculate metrics only for non-noise points
    mask = dbscan_labels != -1
    sil_score_dbscan = silhouette_score(X_train_scaled[mask], dbscan_labels[mask])
    print(f"Silhouette Score (non-noise): {sil_score_dbscan:.4f}")

# COMMAND ----------

# Visualize DBSCAN clusters
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='Spectral', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title('DBSCAN Clustering - PCA Projection', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster (-1 = Noise)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##  11. Hierarchical Clustering

# COMMAND ----------

from sklearn.cluster import AgglomerativeClustering

# Fit Hierarchical Clustering
n_clusters_hier = 3
hierarchical = AgglomerativeClustering(n_clusters=n_clusters_hier, linkage='ward')
hier_labels = hierarchical.fit_predict(X_train_scaled)

# Evaluation
sil_score_hier = silhouette_score(X_train_scaled, hier_labels)
db_score_hier = davies_bouldin_score(X_train_scaled, hier_labels)

print("="*80)
print(f"HIERARCHICAL CLUSTERING (n_clusters={n_clusters_hier}, linkage='ward')")
print("="*80)
print(f"Silhouette Score: {sil_score_hier:.4f}")
print(f"Davies-Bouldin Index: {db_score_hier:.4f}")

# COMMAND ----------

# Visualize hierarchical clusters
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap='tab10', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title(f'Hierarchical Clustering (n={n_clusters_hier}) - PCA Projection', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 🤖 PART 4: Supervised Learning

# COMMAND ----------

# MAGIC %md
# MAGIC ##  12. Initialize MLflow Experiment

# COMMAND ----------

# Set MLflow experiment
EXPERIMENT_NAME = "/Users/fatimaezzahraaboutaleb5@gmail.com/telco-churn-ml-databricks" 
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"✓ MLflow experiment set: {EXPERIMENT_NAME}")
print(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##  13. Train Base Models

# COMMAND ----------

# Helper function for model evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = model.decision_function(X_test)
    
    metrics = {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    return metrics, y_test_pred, y_test_proba

# Storage for results
all_metrics = []
all_predictions = {}
all_roc_data = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.1 Logistic Regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

with mlflow.start_run(run_name="Logistic_Regression"):
    print("Training Logistic Regression...")
    
    # Parameters
    C = 1.0
    max_iter = 1000
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    
    # Train
    lr_model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(lr_model, X_train_scaled, y_train, 
                                              X_test_scaled, y_test, "Logistic Regression")
    
    # Log metrics
    for metric_name, value in metrics.items():
        if metric_name != 'model_name':
            mlflow.log_metric(metric_name, value)
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
    
    # Store results
    all_metrics.append(metrics)
    all_predictions['Logistic Regression'] = (y_test, y_pred)
    all_roc_data['Logistic Regression'] = (y_test, y_proba)
    
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ F1-Score: {metrics['test_f1']:.4f}")
    print(f"✓ ROC-AUC: {metrics['test_roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.2 Support Vector Machine (SVM)

# COMMAND ----------

from sklearn.svm import SVC

with mlflow.start_run(run_name="SVM_RBF"):
    print("Training SVM (RBF kernel)...")
    
    # Parameters
    kernel = 'rbf'
    C = 1.0
    gamma = 'scale'
    mlflow.log_param("kernel", kernel)
    mlflow.log_param("C", C)
    mlflow.log_param("gamma", gamma)
    
    # Train
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(svm_model, X_train_scaled, y_train,
                                              X_test_scaled, y_test, "SVM (RBF)")
    
    # Log metrics
    for metric_name, value in metrics.items():
        if metric_name != 'model_name':
            mlflow.log_metric(metric_name, value)
    
    # Log model
    mlflow.sklearn.log_model(svm_model, "svm_model")
    
    # Store results
    all_metrics.append(metrics)
    all_predictions['SVM (RBF)'] = (y_test, y_pred)
    all_roc_data['SVM (RBF)'] = (y_test, y_proba)
    
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ F1-Score: {metrics['test_f1']:.4f}")
    print(f"✓ ROC-AUC: {metrics['test_roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.3 K-Nearest Neighbors (KNN)

# COMMAND ----------

from sklearn.neighbors import KNeighborsClassifier

with mlflow.start_run(run_name="KNN"):
    print("Training K-Nearest Neighbors...")
    
    # Parameters
    n_neighbors = 5
    weights = 'uniform'
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("weights", weights)
    
    # Train
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    knn_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(knn_model, X_train_scaled, y_train,
                                              X_test_scaled, y_test, "KNN (k=5)")
    
    # Log metrics
    for metric_name, value in metrics.items():
        if metric_name != 'model_name':
            mlflow.log_metric(metric_name, value)
    
    # Log model
    mlflow.sklearn.log_model(knn_model, "knn_model")
    
    # Store results
    all_metrics.append(metrics)
    all_predictions['KNN (k=5)'] = (y_test, y_pred)
    all_roc_data['KNN (k=5)'] = (y_test, y_proba)
    
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ F1-Score: {metrics['test_f1']:.4f}")
    print(f"✓ ROC-AUC: {metrics['test_roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##  14. Tree-Based Models

# COMMAND ----------

# MAGIC %md
# MAGIC ### 14.1 Decision Tree

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

with mlflow.start_run(run_name="Decision_Tree"):
    print("Training Decision Tree...")
    
    # Parameters
    max_depth = 5
    min_samples_split = 50
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    
    # Train
    dt_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(dt_model, X_train_scaled, y_train,
                                              X_test_scaled, y_test, "Decision Tree")
    
    # Log metrics
    for metric_name, value in metrics.items():
        if metric_name != 'model_name':
            mlflow.log_metric(metric_name, value)
    
    # Log model
    mlflow.sklearn.log_model(dt_model, "decision_tree_model")
    
    # Store results
    all_metrics.append(metrics)
    all_predictions['Decision Tree'] = (y_test, y_pred)
    all_roc_data['Decision Tree'] = (y_test, y_proba)
    
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ F1-Score: {metrics['test_f1']:.4f}")
    print(f"✓ ROC-AUC: {metrics['test_roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##  15. Ensemble Methods

# COMMAND ----------

# MAGIC %md
# MAGIC ### 15.1 Random Forest (Bagging)

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run(run_name="Random_Forest"):
    print("Training Random Forest...")
    
    # Parameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("ensemble_method", "Bagging")
    
    # Train
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(rf_model, X_train_scaled, y_train,
                                              X_test_scaled, y_test, "Random Forest")
    
    # Log metrics
    for metric_name, value in metrics.items():
        if metric_name != 'model_name':
            mlflow.log_metric(metric_name, value)
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    
    # Store results
    all_metrics.append(metrics)
    all_predictions['Random Forest'] = (y_test, y_pred)
    all_roc_data['Random Forest'] = (y_test, y_proba)
    
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ F1-Score: {metrics['test_f1']:.4f}")
    print(f"✓ ROC-AUC: {metrics['test_roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 15.2 AdaBoost (Boosting)

# COMMAND ----------

from sklearn.ensemble import AdaBoostClassifier

with mlflow.start_run(run_name="AdaBoost"):
    print("Training AdaBoost...")
    
    # Parameters
    n_estimators = 50
    learning_rate = 1.0
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("ensemble_method", "Boosting")
    
    # Train
    ada_model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    ada_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(ada_model, X_train_scaled, y_train,
                                              X_test_scaled, y_test, "AdaBoost")
    
    # Log metrics
    for metric_name, value in metrics.items():
        if metric_name != 'model_name':
            mlflow.log_metric(metric_name, value)
    
    # Log model
    mlflow.sklearn.log_model(ada_model, "adaboost_model")
    
    # Store results
    all_metrics.append(metrics)
    all_predictions['AdaBoost'] = (y_test, y_pred)
    all_roc_data['AdaBoost'] = (y_test, y_proba)
    
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ F1-Score: {metrics['test_f1']:.4f}")
    print(f"✓ ROC-AUC: {metrics['test_roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 15.3 Gradient Boosting

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier

with mlflow.start_run(run_name="Gradient_Boosting"):
    print("Training Gradient Boosting...")
    
    # Parameters
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("ensemble_method", "Boosting")
    
    # Train
    gb_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                         max_depth=max_depth, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(gb_model, X_train_scaled, y_train,
                                              X_test_scaled, y_test, "Gradient Boosting")
    
    # Log metrics
    for metric_name, value in metrics.items():
        if metric_name != 'model_name':
            mlflow.log_metric(metric_name, value)
    
    # Log model
    mlflow.sklearn.log_model(gb_model, "gradient_boosting_model")
    
    # Store results
    all_metrics.append(metrics)
    all_predictions['Gradient Boosting'] = (y_test, y_pred)
    all_roc_data['Gradient Boosting'] = (y_test, y_proba)
    
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ F1-Score: {metrics['test_f1']:.4f}")
    print(f"✓ ROC-AUC: {metrics['test_roc_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 15.4 XGBoost

# COMMAND ----------

try:
    import xgboost as xgb
    
    with mlflow.start_run(run_name="XGBoost"):
        print("Training XGBoost...")
        
        # Parameters
        n_estimators = 100
        learning_rate = 0.1
        max_depth = 3
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("ensemble_method", "Boosting")
        
        # Train
        xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                     max_depth=max_depth, random_state=42, eval_metric='logloss')
        xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        metrics, y_pred, y_proba = evaluate_model(xgb_model, X_train_scaled, y_train,
                                                  X_test_scaled, y_test, "XGBoost")
        
        # Log metrics
        for metric_name, value in metrics.items():
            if metric_name != 'model_name':
                mlflow.log_metric(metric_name, value)
        
        # Log model
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        
        # Store results
        all_metrics.append(metrics)
        all_predictions['XGBoost'] = (y_test, y_pred)
        all_roc_data['XGBoost'] = (y_test, y_proba)
        
        print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"✓ F1-Score: {metrics['test_f1']:.4f}")
        print(f"✓ ROC-AUC: {metrics['test_roc_auc']:.4f}")
        
except ImportError:
    print("⚠ XGBoost not available. Install with: %pip install xgboost")

# COMMAND ----------

# MAGIC %md
# MAGIC # PART 5: Neural Network (Deep Learning)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Build and Train Neural Network

# COMMAND ----------

# DBTITLE 1,Neural Network Model (Fixed Layer Name)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    with mlflow.start_run(run_name="Neural_Network"):
        print("\nBuilding Neural Network...")
        
        # Parameters
        hidden_layers = [128, 64, 32]
        dropout_rate = 0.3
        learning_rate = 0.001
        epochs = 50
        batch_size = 32
        
        mlflow.log_param("hidden_layers", str(hidden_layers))
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        # Build model
        model = Sequential(name='Churn_Prediction_NN')
        
        # Input layer
        model.add(Dense(hidden_layers[0], activation='relu', input_dim=X_train_scaled.shape[1], name='input_layer_1'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for idx, units in enumerate(hidden_layers[1:], start=2):
            model.add(Dense(units, activation='relu', name=f'hidden_layer_{idx}'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid', name='output_layer'))
        
        # Compile
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy',
                             keras.metrics.Precision(name='precision'),
                             keras.metrics.Recall(name='recall'),
                             keras.metrics.AUC(name='auc')])
        
        print("\n" + "="*80)
        model.summary()
        print("="*80 + "\n")
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        
        # Train
        print("Training Neural Network...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        nn_metrics = {
            'model_name': 'Neural Network',
            'train_accuracy': history.history['accuracy'][-1],
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics
        for metric_name, value in nn_metrics.items():
            if metric_name != 'model_name':
                mlflow.log_metric(metric_name, value)
        
        # Log model
        mlflow.tensorflow.log_model(model, "neural_network_model")
        
        # Store results
        all_metrics.append(nn_metrics)
        all_predictions['Neural Network'] = (y_test, y_pred)
        all_roc_data['Neural Network'] = (y_test, y_pred_proba)
        
        print("\n" + "="*80)
        print("NEURAL NETWORK RESULTS")
        print("="*80)
        print(f"Test Accuracy:  {nn_metrics['test_accuracy']:.4f}")
        print(f"Precision:      {nn_metrics['test_precision']:.4f}")
        print(f"Recall:         {nn_metrics['test_recall']:.4f}")
        print(f"F1-Score:       {nn_metrics['test_f1']:.4f}")
        print(f"ROC-AUC:        {nn_metrics['test_roc_auc']:.4f}")
        print("="*80)
        
        # Plot training history
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Model Loss', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Model Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # AUC
        axes[2].plot(history.history['auc'], label='Training AUC', linewidth=2)
        axes[2].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('AUC', fontsize=12)
        axes[2].set_title('Model AUC', fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
except ImportError as e:
    print(f"⚠ TensorFlow not available: {e}")
    print("Install with: %pip install tensorflow")


# COMMAND ----------

# MAGIC %md
# MAGIC # PART 6: Model Comparison & Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Final Model Comparison

# COMMAND ----------

# Create comparison DataFrame
df_comparison = pd.DataFrame(all_metrics)
df_comparison = df_comparison.sort_values('test_accuracy', ascending=False).reset_index(drop=True)
df_comparison.insert(0, 'Rank', range(1, len(df_comparison) + 1))

print("\n" + "="*120)
print("FINAL MODEL COMPARISON - ALL ALGORITHMS")
print("="*120)
display(df_comparison)
print("="*120)

# Best model
best_model = df_comparison.iloc[0]
print(f"\nBEST MODEL: {best_model['model_name']}")
print(f"   Test Accuracy: {best_model['test_accuracy']:.4f}")
print(f"   F1-Score: {best_model['test_f1']:.4f}")
print(f"   ROC-AUC: {best_model['test_roc_auc']:.4f}")
print("="*120 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Visualizations

# COMMAND ----------

# MAGIC %md
# MAGIC ### 18.1 ROC Curves Comparison

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.tab10(np.linspace(0, 1, len(all_roc_data)))

for idx, (model_name, (y_true, y_proba)) in enumerate(all_roc_data.items()):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ax.plot(fpr, tpr, linewidth=2.5, color=colors[idx],
           label=f'{model_name} (AUC = {roc_auc:.3f})')

# Diagonal line
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves - All Models Comparison', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 18.2 Performance Metrics Bar Charts

# COMMAND ----------

metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
    ax = axes[idx]
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_comparison)))
    
    bars = ax.barh(df_comparison['model_name'], df_comparison[metric], color=colors_gradient)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel(name, fontsize=12)
    ax.set_title(f'{name} Comparison', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 18.3 Confusion Matrices

# COMMAND ----------

n_models = len(all_predictions)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
axes = axes.flatten() if n_models > 1 else [axes]

for idx, (model_name, (y_true, y_pred)) in enumerate(all_predictions.items()):
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Churn', 'Churn'],
               yticklabels=['No Churn', 'Churn'],
               ax=axes[idx], cbar=True)
    
    axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=11)
    axes[idx].set_xlabel('Predicted Label', fontsize=11)

# Hide unused subplots
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 18.4 Feature Importance (for tree-based models)

# COMMAND ----------

# Plot feature importance for Random Forest
try:
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_gradient = plt.cm.plasma(np.linspace(0.2, 0.9, len(indices)))
    
    feature_names_list = X.columns.tolist()
    top_features = [feature_names_list[i] for i in indices]
    top_importances = importances[indices]
    
    bars = ax.barh(top_features, top_importances, color=colors_gradient)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
except:
    print("Feature importance plot skipped (Random Forest not trained)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Export Results

# COMMAND ----------

# Save comparison table to CSV
output_path = "/Workspace/Users/fatimaezzahraaboutaleb5@gmail.com/telco-churn-ml-databricks/comparison.csv"
df_comparison.to_csv(output_path, index=False)
print(f"✓ Results exported to: {output_path}")

# Log to MLflow
mlflow.log_artifact(output_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC **Author:** Fatima-Ezzahra ABOUTALEB  
# MAGIC **Course:** Machine Learning  
# MAGIC **Date:** January 2026
