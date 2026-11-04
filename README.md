# Customer Segmentation — K-Means Clustering 🎯👥

An unsupervised machine learning project grouping customer bases into distinct behavioral cohorts using **K-Means Clustering** and the **Elbow Method**.

---

## 📌 Problem Overview
Targeted marketing requires discovering hidden customer groupings without prior ground-truth labels. This model clusters users along two core behavioral dimensions:
1. **Total Spent ($)**: Financial volume per account.
2. **Purchase Frequency**: Total transaction cadence.

Features are normalized using `StandardScaler` to prevent feature dominance before clustering.

---

## 🔍 Methodology
1. **Feature Standardization:** Zero-mean, unit-variance transformation:
   $$z = \frac{x - \mu}{\sigma}$$
2. **Elbow Method:** Evaluates Within-Cluster Sum of Squares (WCSS / Inertia) across $k \in [1, 10]$ to find the point of diminishing returns.
3. **Cluster Assignment:** Segments accounts into high-value loyalists, frequent small buyers, and occasional shoppers.

---

## 🛠 Tech Stack
* **Python 3.x**
* **scikit-learn** (`KMeans`, `StandardScaler`)
* **Pandas & NumPy**
* **Matplotlib** (WCSS Elbow curve & cluster scatter visualization)

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/Devanshu155/ARKA_ML_02.git
cd ARKA_ML_02
pip install -r requirements.txt
```

### 2. Execution
```bash
python task2.py
```

---

## 👤 Author
**Devanshu Mishra** ([@Devanshu155](https://github.com/Devanshu155))
