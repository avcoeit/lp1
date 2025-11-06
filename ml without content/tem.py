import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


mall_data = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\lp-1\Mall_Customers.csv")

#mall_data = pd.read_csv("Mall_Customers.csv") #keep csv in same folder of program
#mall_data = pd.read_csv("/home/avcoe/Mall_Customers.csv") for ubuntu
mall_data.head()
mall_data.info()

corr = mall_data.corr(numeric_only=True)
plt.figure(figsize=(8,8))
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,cmap='Reds')

plt.figure(figsize=(10,10))
sns.countplot(x="Gender", data=mall_data)

plt.figure(figsize=(16,10))
sns.countplot(x="Age", data=mall_data)

plt.figure(figsize=(20,8))
sns.barplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=mall_data)

X = mall_data.iloc[:,[2,3,4]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=50)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
wcss

sns.set()
plt.plot(range(1,11),wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS value")
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++',random_state = 0)
y = kmeans.fit_predict(X)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 0,0],X[y == 0,1],X[y == 0,2], s = 40 , color = 'red', label = "cluster 1")
ax.scatter(X[y == 1,0],X[y == 1,1],X[y == 1,2], s = 40 , color = 'blue', label = "cluster 2")
ax.scatter(X[y == 2,0],X[y == 2,1],X[y == 2,2], s = 40 , color = 'green', label = "cluster 3")
ax.scatter(X[y == 3,0],X[y == 3,1],X[y == 3,2], s = 40 , color = 'yellow', label = "cluster 4")
ax.scatter(X[y == 4,0],X[y == 4,1],X[y == 4,2], s = 40 , color = 'purple', label = "cluster 5")
ax.set_xlabel('Age of a customer-->')
ax.set_ylabel('Anual Income-->')
ax.set_zlabel('Spending Score-->')
ax.legend()
plt.show()


"""
----------------------------- EXPLANATION -----------------------------

1️⃣ Importing Libraries:
   - numpy, pandas → for numerical and tabular data handling
   - matplotlib.pyplot, seaborn → for data visualization
   - sklearn.cluster.KMeans → for applying the K-Means clustering algorithm

2️⃣ Loading Dataset:
   - The dataset "Mall_Customers.csv" is read into a pandas DataFrame.
   - mall_data.head() displays the first few rows.
   - mall_data.info() shows dataset info such as column names, non-null counts, and data types.

3️⃣ Correlation Heatmap:
   - Computes correlation among numeric columns.
   - The heatmap visually shows how features relate to each other.

4️⃣ Gender and Age Analysis:
   - Count plots display which gender shops more and which age group has more customers.

5️⃣ Income vs Spending Analysis:
   - A bar plot shows how annual income relates to spending scores, giving insight into purchasing behavior.

6️⃣ Selecting Features for Clustering:
   - Chooses 'Age', 'Annual Income (k$)', and 'Spending Score (1–100)' for clustering (columns 2, 3, 4).

7️⃣ Finding Optimal Clusters using WCSS:
   - WCSS (Within Cluster Sum of Squares) is calculated for cluster counts from 1 to 10.
   - The elbow graph helps identify the best number of clusters (around 5 here).

8️⃣ Building K-Means Model:
   - Creates a KMeans model with 5 clusters.
   - Fits the model on selected features and assigns each customer a cluster label (0–4).

9️⃣ 3D Visualization:
   - Plots clusters in a 3D scatterplot with different colors for each cluster.
   - X-axis → Age
   - Y-axis → Annual Income
   - Z-axis → Spending Score

✅ OUTPUT:
   - Displays several visualizations (heatmap, bar plots, scatterplots).
   - Groups customers into 5 clusters based on their purchasing patterns.

----------------------------- INSTALLATION GUIDE -----------------------------

To install all required libraries in Spyder (via Anaconda Prompt):

conda install pandas numpy matplotlib seaborn scikit-learn

OR using pip:

pip install pandas numpy matplotlib seaborn scikit-learn

-----------------------------------------------------------------------
Run each line in Spyder using **F9** to view output step-by-step.
-----------------------------------------------------------------------
"""
