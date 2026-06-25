# 🍽️💰 Restaurant Sentiment & Customer Financial Segmentation

Two end-to-end Machine Learning projects covering both supervised NLP classification and unsupervised clustering.

---

## 📁 Project 1 — Restaurant Review Sentiment Analysis

### Overview
Binary text classification model that predicts whether a restaurant review is **Positive** or **Negative** based on its content.

### Pipeline
- **Data Cleaning** — removed duplicates, handled missing values, fixed rating types
- **Text Preprocessing** — lowercasing, regex cleaning, noise removal
- **Feature Extraction** — TF-IDF Vectorizer (unigrams + bigrams, 20K features)
- **Models Trained**
  - Logistic Regression (class_weight=balanced)
  - LinearSVC + CalibratedClassifierCV for probability outputs
- **Evaluation** — Classification Report, Confusion Matrix, ROC-AUC Curve

### Saved Artifacts
| File | Description |
|------|-------------|
| `tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer |
| `logistic_classweight_balanced.joblib` | Trained Logistic Regression model |
| `linear_svc_model.joblib` | Trained LinearSVC model |
| `linear_svc_calibrated_model.joblib` | Calibrated SVC with probability support |

---

## 📁 Project 2 — Customer Financial Segmentation

### Overview
Unsupervised clustering pipeline that segments bank customers into distinct financial profiles based on their banking behavior and financial features.

### Pipeline
- **EDA** — distributions, boxplots, missing values, duplicates
- **Outlier Handling** — IQR capping
- **Feature Engineering**
  - `Total_Banking_Products`
  - `Income_per_Person`
  - `Spending_Income_Ratio`
  - `Has_Mortgage_Flag`
- **Dimensionality Reduction** — PCA (90% variance threshold)
- **Models**
  - K-Means — tuned via Silhouette Score sweep (k = 2–15)
  - DBSCAN — grid search over `eps` and `min_samples`
- **Evaluation** — Silhouette Score, Davies-Bouldin Index

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243)

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` — TF-IDF, Logistic Regression, LinearSVC, KMeans, DBSCAN, PCA
- `joblib` — model serialization

---

## 📂 Repository Structure
