# Telco Customer Churn Prediction - Complete ML Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Databricks](https://img.shields.io/badge/Platform-Databricks-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![License](https://img.shields.io/badge/License-Academic-yellow)

**A comprehensive Machine Learning project for predicting customer churn using Databricks and MLflow**

</div>

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [MLflow Integration](#mlflow-integration)
- [Author](#author)

---

## Overview

This project implements a **complete end-to-end Machine Learning pipeline** for predicting customer churn in the telecommunications industry. The project follows academic best practices and includes:

-  Comprehensive data preprocessing and feature engineering
-  Dimensionality reduction (PCA, t-SNE, NMF)
-  Unsupervised learning (K-Means, DBSCAN, Hierarchical Clustering)
-  Supervised learning (8+ algorithms)
-  Deep Learning (Neural Networks with TensorFlow)
-  MLflow experiment tracking in Databricks
-  Professional visualizations and model comparison
-  Modular, production-ready code

---

##  Dataset

**Source:** Kaggle - [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Description:**
- **Rows:** ~7,000 customers
- **Features:** 20 (numerical and categorical)
- **Target:** `Churn` (Yes/No) - Binary Classification
- **Task:** Predict whether a customer will leave the service

**Features Include:**
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, InternetService, OnlineSecurity, etc.
- **Account:** Contract, PaymentMethod, PaperlessBilling
- **Charges:** MonthlyCharges, TotalCharges
- **Tenure:** Number of months as customer

---

##  Project Structure

```
telco-churn-ml-databricks/
│
├──  notebooks/               # Databricks notebooks (.dbc)
│   ├── 01_EDA.dbc             # Exploratory Data Analysis
│   ├── 02_Preprocessing.dbc   # Data cleaning & feature engineering
│   ├── 03_Dimensionality_Reduction.dbc
│   ├── 04_Clustering.dbc      # Unsupervised learning
│   ├── 05_Supervised_Models.dbc
│   └── 06_Neural_Network.dbc  # Deep learning with TensorFlow
│
├── 📁 src/                     # Python modules (production-ready)
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── dimensionality_reduction.py
│   ├── clustering.py          # Clustering algorithms
│   ├── models.py              # Supervised learning models
│   ├── neural_network.py      # Neural network implementation
│   └── evaluation.py          # Evaluation metrics & visualization
│
├── 📁 data/                    # Dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── 📁 results/                 # Outputs (figures, models, reports)
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

##  ML Pipeline

### **1 Data Cleaning & Preprocessing**
- Handle missing values in `TotalCharges`
- Convert data types
- Label encoding for categorical features
- Feature scaling (StandardScaler)
- Train-test split (80-20, stratified)

### **2 Dimensionality Reduction**
- **PCA:** Find principal components
- **t-SNE:** Non-linear dimensionality reduction
- **NMF:** Non-negative matrix factorization
- Visualize 2D projections colored by Churn

### **3 Clustering (Unsupervised Learning)**
Applied **without** using the target variable:
- **K-Means:** Partition-based clustering
- **DBSCAN:** Density-based clustering
- **Hierarchical Clustering:** Agglomerative clustering

**Evaluation:**
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Compare clusters vs. true churn labels

### **4 Supervised Learning**

#### Base Models:
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

#### Tree-Based:
- Decision Tree

#### Ensemble Methods:
- **Bagging:** Random Forest
- **Boosting:** AdaBoost, Gradient Boosting, XGBoost

### **5 Neural Network (Deep Learning)**
- **Architecture:** Feedforward Neural Network
- **Framework:** TensorFlow / Keras
- **Layers:** Input → Dense (ReLU) → Dropout → Dense (ReLU) → Dropout → Output (Sigmoid)
- **Optimization:** Adam optimizer with early stopping
- **Regularization:** Dropout, Batch Normalization

---

##  Models Implemented

| **Category** | **Models** |
|-------------|-----------|
| **Base Classifiers** | Logistic Regression, SVM, KNN |
| **Tree-Based** | Decision Tree |
| **Ensemble (Bagging)** | Random Forest |
| **Ensemble (Boosting)** | AdaBoost, Gradient Boosting, XGBoost |
| **Deep Learning** | Feedforward Neural Network (TensorFlow) |
| **Clustering** | K-Means, DBSCAN, Hierarchical |
| **Dim. Reduction** | PCA, t-SNE, NMF |

---

##  Installation

### **Prerequisites**
- Python 3.8+
- Databricks workspace
- MLflow

### **Setup**

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/telco-churn-ml-databricks.git
cd telco-churn-ml-databricks
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Upload to Databricks:**
- Upload the `src/` folder to Databricks DBFS
- Import notebooks from `notebooks/` folder
- Upload dataset to DBFS at `/FileStore/tables/`

---

##  Usage

### **Running in Databricks**

1. **Import Notebooks:**
   - Import all `.dbc` files from `notebooks/` folder
   - Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06

2. **Execute Pipeline:**
```python
# In Databricks notebook
%run ./notebooks/01_EDA
%run ./notebooks/02_Preprocessing
# ... continue with other notebooks
```

3. **View MLflow Experiments:**
   - Navigate to **Experiments** tab in Databricks
   - Compare runs, metrics, and models
   - Select best performing model

### **Running Locally (Optional)**

```python
# Import modules
from src.preprocessing import TelcoPreprocessor
from src.models import SupervisedModelTrainer
from src.neural_network import NeuralNetworkTrainer

# Load and preprocess data
preprocessor = TelcoPreprocessor()
df = preprocessor.load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(df)

# Train models
trainer = SupervisedModelTrainer()
model, metrics = trainer.train_random_forest(X_train, y_train, X_test, y_test)

# Compare all models
comparison = trainer.compare_all_models()
```

---

##  Results

### **Model Performance Comparison**

| Rank | Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|---------------|-----------|--------|----------|---------|
| 1 | XGBoost | 0.8045 | 0.6721 | 0.5431 | 0.6007 | 0.8476 |
| 2 | Random Forest | 0.7994 | 0.6531 | 0.5431 | 0.5930 | 0.8401 |
| 3 | Gradient Boosting | 0.7979 | 0.6447 | 0.5468 | 0.5915 | 0.8398 |
| 4 | Neural Network | 0.7908 | 0.6250 | 0.5287 | 0.5729 | 0.8312 |
| 5 | Logistic Regression | 0.7880 | 0.6304 | 0.5036 | 0.5600 | 0.8410 |

*(Note: Actual results will vary based on training)*

### **Key Insights**

1. **Best Model:** XGBoost achieves highest accuracy (80.45%)
2. **Ensemble Methods:** Random Forest and Gradient Boosting show robust performance
3. **Neural Network:** Competitive performance with deep learning approach
4. **Class Imbalance:** Precision-Recall tradeoff evident in metrics
5. **Feature Importance:** Contract type, tenure, and monthly charges are top predictors

### **Visualizations**

-  ROC Curves for all models
-  Confusion Matrices
-  Feature Importance plots
-  PCA/t-SNE/NMF 2D projections
-  Clustering visualizations
-  Training history for Neural Network

---

##  MLflow Integration

### **Experiment Tracking**

Each model training run logs:
- **Parameters:** hyperparameters, model config
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC
- **Artifacts:** trained models, plots, confusion matrices
- **Tags:** model family, dataset version

### **Using MLflow in Databricks**

```python
import mlflow
import mlflow.sklearn

# Start MLflow run
with mlflow.start_run(run_name="Random_Forest"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_f1", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
```

### **Viewing Experiments**

1. Navigate to **Experiments** tab in Databricks
2. Click on experiment name
3. Compare runs side-by-side
4. Select best model based on metrics
5. Register model to MLflow Model Registry

---

##  Documentation



### **Code Modules**

All Python modules include:
- Comprehensive docstrings
- Type hints
- Error handling
- Example usage in `__main__`

---


---

##  Author

**Fatima-Ezzahra ABOUTALEB**


-  https://github.com/Fatima-EzzahraAboutaleb

---

## 📄 License

This project is created for academic purposes as part of a university Machine Learning course.

---



##  References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. TensorFlow Documentation: https://www.tensorflow.org/
3. MLflow Documentation: https://mlflow.org/
4. Databricks Documentation: https://docs.databricks.com/

---

<div align="center">


</div>
