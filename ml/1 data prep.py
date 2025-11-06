
# To install required libraries in Spyder, open Anaconda Prompt and run:
#conda install pandas matplotlib seaborn numpy scikit-learn


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics as s

# Setting visual styles for plots
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# -----------------------------------------------------------
# Load the dataset
# Run this line using F9 to execute current line in Spyder
# -----------------------------------------------------------
df = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\lp-1\heart.csv")
df.head()   # Display first few rows of dataset

# -----------------------------------------------------------
# Display information about dataset columns and data types
# -----------------------------------------------------------
df.info()

# dataframe.size -> gives total number of elements in dataframe
size = df.size
print("Size of dataset is :",size)

# dataframe.shape -> gives (rows, columns)
shape = df.shape
print("shape of datset is \n\n:",shape)

# -----------------------------------------------------------
# Display statistical summary of numeric columns
# -----------------------------------------------------------
print(df.describe())

# Display first 5 rows again
print(df.head())

# -----------------------------------------------------------
# Check data types of all columns
# -----------------------------------------------------------
print("Data Type for Each Columns are\n",df.dtypes.value_counts())

# -----------------------------------------------------------
# Identify object (categorical) columns
# -----------------------------------------------------------
df.dtypes == 'object'

# Selecting numeric columns only
n = df.columns[df.dtypes != 'object']

# Display all numeric columns
df[n]

# -----------------------------------------------------------
# Display missing (null) values in numeric columns
# -----------------------------------------------------------
print("",df[n].isnull())

# -----------------------------------------------------------
# Count total missing values per column
# -----------------------------------------------------------
df[n].isnull().sum().sort_values(ascending=False)

# -----------------------------------------------------------
# Finding percentage of null values in each column
# -----------------------------------------------------------
df[n].isnull().sum().sort_values(ascending=False)/len(df)

# -----------------------------------------------------------
# Display 'age' column
# -----------------------------------------------------------
df['age']

# -----------------------------------------------------------
# Finding Average Age of Patients using statistics module
# -----------------------------------------------------------
average = s.mean(df['age'])
print("Average age : ",average)

# -----------------------------------------------------------
# Extracting Particular Columns
# -----------------------------------------------------------
print(df['age'])
print(df['sex'])
print(df['trestbps'])
print(df['chol'])

# -----------------------------------------------------------
# Displaying Correlation Matrix (Confusion Heatmap)
# -----------------------------------------------------------
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# -----------------------------------------------------------
# Import metrics for model evaluation
# -----------------------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------------------------------
# Function to print accuracy, confusion matrix, and classification report
# -----------------------------------------------------------
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
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print("Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print("Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

# -----------------------------------------------------------
# Split the dataset into training and testing sets
# -----------------------------------------------------------
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)   # Features
X.shape
y = df.target                   # Target variable
y.shape

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

print(X_train)

# -----------------------------------------------------------
# Classification using Logistic Regression
# -----------------------------------------------------------
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)   # Train the model

# Evaluate performance on training and testing data
print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

# -----------------------------------------------------------
# Display Training and Testing Accuracy for Logistic Regression
# -----------------------------------------------------------
test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df

"""
# -----------------------------------------------------------
# Classification using K-nearest neighbors (KNN)
# -----------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)   # Train the KNN model

# Evaluate KNN model
print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)

# Display Training and Testing Accuracy for K-nearest neighbors
test_score = accuracy_score(y_test, knn_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, knn_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df
"""
