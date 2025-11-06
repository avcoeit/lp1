import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

#df = pd.read_csv("Admission_Predict.csv") #keep csv in same folder of code
df = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\lp-1\Admission_Predict.csv")
#df = pd.read_csv("/home/avcoe/Admission_Predict.csv") for ubuntu
df.head()
df.info()

size = df.size
print("Size of dataset is :", size)

shape = df.shape
print("Shape of dataset is :", shape)

print(df.describe())
print(df.head())

print("Data Type for Each Column:\n", df.dtypes.value_counts())

df.dtypes == 'object'
n = df.columns[df.dtypes != 'object']
df[n]

print("Missing values in numeric columns:\n", df[n].isnull())
print("Total Missing Values per Column:\n", df[n].isnull().sum().sort_values(ascending=False))
print("Percentage of Missing Values:\n", df[n].isnull().sum().sort_values(ascending=False) / len(df))

dataset = df.copy()

from sklearn.preprocessing import StandardScaler
s_sc = StandardScaler()

col_to_scale = ['GRE Score', 'CGPA']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

from sklearn.model_selection import train_test_split
X = dataset.drop('target', axis=1)
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

print("Training Accuracy (%):", accuracy_score(y_train, tree_clf.predict(X_train)) * 100)
print("Testing Accuracy (%):", accuracy_score(y_test, tree_clf.predict(X_test)) * 100)

# -----------------------------------------------------------
# 🧠 EXPLANATION OF THE CODE
# -----------------------------------------------------------
# 1. Libraries imported:
#    - pandas: Used for loading and handling the dataset.
#    - matplotlib.pyplot & seaborn: Used for data visualization and setting visual styles.
#    - sklearn: Used for preprocessing, splitting data, and building the Decision Tree model.
#
# 2. Dataset loading:
#    - The dataset 'Admission_Predict.csv' is loaded using pandas.
#    - .head(), .info(), .describe(), .shape, and .size are used to understand its structure.
#
# 3. Data preprocessing:
#    - Checked the datatype of each column and identified numeric columns.
#    - Checked for missing values and calculated the total and percentage of missing data.
#
# 4. Feature scaling:
#    - Scaled 'GRE Score' and 'CGPA' using StandardScaler to normalize values for better model performance.
#
# 5. Splitting dataset:
#    - Data is divided into training (80%) and testing (20%) subsets using train_test_split.
#    - X contains input features, and y contains the target variable.
#
# 6. Model building:
#    - A DecisionTreeClassifier model is created and trained on the training dataset.
#
# 7. Evaluation:
#    - The function 'print_score' calculates and prints the accuracy, confusion matrix, and classification report
#      for both training and testing datasets.
#    - Accuracy scores for training and testing are displayed at the end.
#
# 8. Output:
#    - Displays all important details: dataset summary, model performance, and metrics.
#
# -----------------------------------------------------------
# ⚙️ HOW TO INSTALL REQUIRED LIBRARIES (Run these in Anaconda Prompt or Terminal)
# -----------------------------------------------------------
# conda install pandas matplotlib seaborn numpy scikit-learn
# or (if using pip)
# pip install pandas matplotlib seaborn numpy scikit-learn
#
# -----------------------------------------------------------
# 💡 NOTE:
# - Ensure 'Admission_Predict.csv' is in the same directory as your Python file.
# - Replace 'target' with the actual name of your dependent variable column (e.g., 'Chance of Admit').
# - You can run each section in Spyder using F9.
# -----------------------------------------------------------
