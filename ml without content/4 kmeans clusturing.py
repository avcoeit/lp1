import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Load dataset
mall_data = pd.read_csv("C:/Users/Prathmesh/OneDrive/Desktop/LP/CSV File/Mall_Customers.csv")

# Display first rows
print(mall_data.head())

# Summary info
mall_data.info()

# ============================
#   CORRELATION HEATMAP FIXED
# ============================
corr = mall_data.corr(numeric_only=True)
# Select only numeric data for correlation
corr = mall_data.select_dtypes(include=['int64', 'float64']).corr()

plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot=True, cmap='Reds')
plt.title("Correlation Heatmap")
plt.show()

# ============================
#     GENDER COUNT PLOT
# ============================
plt.figure(figsize=(10,6))
sns.countplot(x="Gender", data=mall_data)
plt.title("Count of Customers by Gender")
plt.show()

# ============================
#     AGE DISTRIBUTION
# ============================
plt.figure(figsize=(16,6))
sns.countplot(x="Age", data=mall_data)
plt.title("Age Distribution of Customers")
plt.show()

# ============================
#   INCOME vs SPENDING SCORE
# ============================
plt.figure(figsize=(18,6))
sns.barplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=mall_data)
plt.title("Annual Income vs Spending Score")
plt.show()

# ============================
#     K-MEANS CLUSTERING
# ============================
X = mall_data.iloc[:, [2, 3, 4]].values   # Age, Annual Income, Spending Score

# Find WCSS (Elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=50)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Elbow plot
plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method For Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Build model with optimal clusters (5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=50)
y = kmeans.fit_predict(X)

# ============================
#     3D CLUSTER PLOT
# ============================
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], s = 40, color = 'red', label = "Cluster 1")
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], s = 40, color = 'blue', label = "Cluster 2")
ax.scatter(X[y == 2, 0], X[y == 2, 1], X[y == 2, 2], s = 40, color = 'green', label = "Cluster 3")
ax.scatter(X[y == 3, 0], X[y == 3, 1], X[y == 3, 2], s = 40, color = 'yellow', label = "Cluster 4")
ax.scatter(X[y == 4, 0], X[y == 4, 1], X[y == 4, 2], s = 40, color = 'purple', label = "Cluster 5")

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
ax.set_title("Customer Segmentation (3D Clusters)")
ax.legend()

plt.show()
