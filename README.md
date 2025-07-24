
# 🛍️ Customer Segmentation using K-Means and DBSCAN

### 📊 Unsupervised Learning Project | Elevvo Pathways

This project clusters mall customers into meaningful segments using **K-Means** and **DBSCAN** algorithms. It explores purchasing behavior by analyzing **Annual Income** and **Spending Score**, helping businesses target customers effectively.

---

## 📁 Dataset

* **Dataset Used:** Mall Customer Dataset (Kaggle)
* **Attributes:**

  * `CustomerID`
  * `Gender`
  * `Age`
  * `Annual Income (k$)`
  * `Spending Score (1–100)`

---

## 🧰 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
```

---

## 🔍 Step-by-Step Explanation

### 1. Loading and Exploring the Data

Loaded the dataset using `pd.read_csv()` and viewed its structure using `head()`, `info()`, and `describe()`.

### 2. Check for Missing Values

Checked for null values using `df.isnull().sum()` to ensure data cleanliness.

### 3. Data Visualization

Plotted histograms for `Age` and `Annual Income` and a count plot for `Gender` to understand data distribution.

### 4. Feature Scaling

Used Min-Max Scaling on `Annual Income` and `Spending Score` to normalize features between 0 and 1 using:

```python
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
```

### 5. Determine Optimal K using Elbow Method

Computed WCSS (within-cluster sum of squares) for K=1 to 10 and visualized it to find the “elbow” point:

```python
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(df_scaled)
    wcss.append(km.inertia_)
```

### 6. Apply K-Means Clustering

Applied K-Means with `n_clusters=5` and added a new `cluster` column to the DataFrame.

### 7. Visualize K-Means Clusters

Plotted a scatter plot with color-coded clusters and marked centroids using:

```python
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='*')
```

### 8. Pie Chart of Income by Cluster

Added the `cluster` column back to the original DataFrame and used Plotly to show the income distribution by cluster.

### 9. Apply DBSCAN Clustering

Used DBSCAN with `eps=0.1` and `min_samples=4` on the scaled data. Added `dbscan_cluster` column to capture labels (clusters & outliers).

### 10. Visualize DBSCAN Results

Plotted DBSCAN clusters with a scatter plot, where outliers (label `-1`) were highlighted differently.

### 11. Compare K-Means vs DBSCAN

Displayed both K-Means and DBSCAN results side by side using subplots for better comparative understanding.

### 12. Average Spending Score per Cluster

Grouped data by `cluster` and calculated average spending scores for each segment to evaluate profitability.

---

## 📌 Key Takeaways

* **K-Means** efficiently segments customers into clear, well-separated clusters.
* **DBSCAN** is robust to outliers and non-spherical shapes but requires careful parameter tuning.
* Combining both techniques offers a well-rounded view of customer behavior and segmentation.

---

## 📈 Output Visuals

* Elbow Curve for K-Selection
* KMeans Cluster Plot with Centroids
* Pie Chart of Annual Income per Cluster
* DBSCAN Cluster Plot
* Side-by-Side Clustering Comparison

---

## 🧠 Concepts Covered

* Clustering
* Feature Scaling
* Elbow Method
* K-Means
* DBSCAN
* Unsupervised Learning
* Outlier Detection
* Data Visualization

---

## ✅ Project Completed Under Elevvo Pathways

This project was successfully completed as part of the **Elevvo Pathways – Machine Learning Track**, enhancing hands-on skills in unsupervised learning, data visualization, and clustering techniques using real-world customer data.

---

