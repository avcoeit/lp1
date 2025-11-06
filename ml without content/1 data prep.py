import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics as s

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

df = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\lp-1\heart.csv")
#df = pd.read_csv("/home/avcoe/heart.csv") for ubuntu
df.head()
df.info()
size = df.size
print("Size of dataset is :", size)
shape = df.shape
print("shape of datset is \n\n:", shape)
print(df.describe())
print(df.head())
print("Data Type for Each Columns are\n", df.dtypes.value_counts())
df.dtypes == 'object'
n = df.columns[df.dtypes != 'object']
df[n]
print("", df[n].isnull())
df[n].isnull().sum().sort_values(ascending=False)
df[n].isnull().sum().sort_values(ascending=False) / len(df)
df['age']
average = s.mean(df['age'])
print("Average age : ", average)
print(df['age'])
print(df['sex'])
print(df['trestbps'])
print(df['chol'])

corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print("Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print("Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    elif train == False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print("Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print("Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
X.shape
y = df.target
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
print(X_train)

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]],
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df

"""
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)

test_score = accuracy_score(y_test, knn_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, knn_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", train_score, test_score]],
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df
"""

"""
----------------------------- DETAILED EXPLANATION -----------------------------

1️⃣ Importing Libraries:
   - pandas, numpy → data handling and numerical operations
   - matplotlib, seaborn → data visualization
   - statistics → used to calculate mean (average)
   - sklearn → machine learning models and evaluation metrics

2️⃣ Loading Dataset:
   df = pd.read_csv("heart.csv") loads the heart disease dataset into a DataFrame.

3️⃣ Dataset Overview:
   - df.head() → shows first 5 rows
   - df.info() → shows column names, types, and missing values
   - df.size → total elements
   - df.shape → (rows, columns)
   - df.describe() → statistical summary
   - df.dtypes.value_counts() → shows how many columns are numeric or object type

4️⃣ Missing Value Analysis:
   - df[n].isnull() → checks missing data (True/False)
   - df[n].isnull().sum() → total missing values per column
   - df[n].isnull().sum()/len(df) → % of missing values per column

5️⃣ Basic Data Insights:
   - df['age'] → selects 'age' column
   - s.mean(df['age']) → computes average age

6️⃣ Correlation Heatmap:
   sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu") → shows correlation between variables.
   High values mean strong relationships between columns.

7️⃣ Machine Learning Preparation:
   - X = df.drop('target', axis=1) → features
   - y = df.target → target output
   - train_test_split(...) → splits data into 80% training and 20% testing.

8️⃣ Logistic Regression Model:
   lr_clf = LogisticRegression(solver='liblinear') → creates model
   lr_clf.fit(X_train, y_train) → trains model
   print_score() → prints accuracy, classification report, and confusion matrix.

9️⃣ Model Evaluation:
   - accuracy_score() → measures prediction accuracy.
   - train_score & test_score → training and testing accuracy in percentage.
   - results_df → DataFrame that stores model name and accuracies.

🔟 Optional KNN Model:
   Uncomment the last section to test K-Nearest Neighbors (KNN) and compare results.

⚙️ Library Installation Commands (Anaconda Prompt):
   conda install pandas matplotlib seaborn numpy scikit-learn

   or using pip:
   pip install pandas matplotlib seaborn numpy scikit-learn

-----------------------------------------------------------------------------
"""
