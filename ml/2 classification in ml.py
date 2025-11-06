# -----------------------------------------------------------
# 🛠️ Install required libraries in Anaconda Prompt before running:
# conda install pandas matplotlib seaborn numpy scikit-learn
# -----------------------------------------------------------

# Importing necessary libraries
import pandas as pd                # For handling datasets (data loading, manipulation)
import matplotlib.pyplot as plt     # For data visualization (graphs, plots)
import seaborn as sns               # For advanced visualizations
sns.set_style("whitegrid")          # Set seaborn style
plt.style.use("fivethirtyeight")    # Set matplotlib style

# -----------------------------------------------------------
# Load the dataset
# NOTE: Ensure 'Admission_Predict.csv' is in the same folder as your Python file.
# OR use the full path, for example:
# df = pd.read_csv("C:/Users/Hp/OneDrive/Desktop/lp-1/Admission_Predict.csv")
# -----------------------------------------------------------
df = pd.read_csv("Admission_Predict.csv")

# Display the first few rows of the dataset
df.head()  # (Run using F9 to see output)

# -----------------------------------------------------------
# 🧾 Basic Information about the dataset
# -----------------------------------------------------------
df.info()  # Shows column names, data types, and missing values

# Get total number of elements (rows × columns)
size = df.size
print("Size of dataset is :", size)

# Get shape (rows, columns)
shape = df.shape
print("Shape of dataset is :", shape)

# -----------------------------------------------------------
# 📊 Statistical summary of numerical columns
# -----------------------------------------------------------
print(df.describe())

# Display first few rows again for reference
print(df.head())

# -----------------------------------------------------------
# 🔎 Data type analysis
# -----------------------------------------------------------
print("Data Type for Each Column:\n", df.dtypes.value_counts())

# Check which columns are NOT of type 'object' (numerical)
df.dtypes == 'object'
n = df.columns[df.dtypes != 'object']  # Select numeric columns

# Display all numeric values
df[n]

# -----------------------------------------------------------
# 🚨 Check for Missing Values
# -----------------------------------------------------------
print("Missing values in numeric columns:\n", df[n].isnull())

# Count of missing values in each column
print("Total Missing Values per Column:\n", df[n].isnull().sum().sort_values(ascending=False))

# Percentage of missing values in each column
print("Percentage of Missing Values:\n", df[n].isnull().sum().sort_values(ascending=False) / len(df))

# -----------------------------------------------------------
# 🧮 Feature Scaling (Normalization)
# -----------------------------------------------------------
dataset = df.copy()

from sklearn.preprocessing import StandardScaler
s_sc = StandardScaler()

# Select columns to scale
col_to_scale = ['GRE Score', 'CGPA']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

# -----------------------------------------------------------
# ✂️ Split the dataset into training and testing data
# -----------------------------------------------------------
from sklearn.model_selection import train_test_split

# NOTE: Replace 'target' with your actual target column name (for example, 'Chance of Admit')
X = dataset.drop('target', axis=1)  # Features
y = dataset.target                  # Target variable

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------
# 🌳 Decision Tree Classifier
# -----------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier

# Create model
tree_clf = DecisionTreeClassifier(random_state=42)

# Train model
tree_clf.fit(X_train, y_train)

# -----------------------------------------------------------
# 📈 Model Evaluation
# -----------------------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define function to print training & testing performance
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n{confusion_matrix(y_train, pred)}\n")
        
    else:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, pred)}\n")

# Print training and testing results
print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

# -----------------------------------------------------------
# 🎯 Final Accuracy Scores
# -----------------------------------------------------------
print("Training Accuracy (%):", accuracy_score(y_train, tree_clf.predict(X_train)) * 100)
print("Testing Accuracy (%):", accuracy_score(y_test, tree_clf.predict(X_test)) * 100)

# -----------------------------------------------------------
# ✅ SUMMARY
# -----------------------------------------------------------
# 1️⃣ Data loaded and basic info extracted (shape, types, missing values)
# 2️⃣ Scaled key features ('GRE Score' and 'CGPA') using StandardScaler
# 3️⃣ Split dataset into train (80%) and test (20%)
# 4️⃣ Trained a Decision Tree Classifier
# 5️⃣ Evaluated model using accuracy, confusion matrix, and classification report
# -----------------------------------------------------------
