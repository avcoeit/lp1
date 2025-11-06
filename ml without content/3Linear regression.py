import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\lp-1\temperatures.csv")
#dataset= pd.read_csv('/home/avcoe/temperatures.csv')  for ubuntu

print(dataset.shape)
print(dataset.describe())

dataset.plot(x='JAN', y='FEB', style='o')
plt.title('JAN vs FEB')
plt.xlabel('mintemp')
plt.ylabel('maxtemp')
plt.show()

plt.figure(figsize=(15, 10))
plt.tight_layout()
seaborn.distplot(dataset['FEB'])
plt.show()

X = dataset['JAN'].values.reshape(-1, 1)
y = dataset['FEB'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

print('Intercept is :', model.intercept_)
print('Coefficient is :', model.coef_)

y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print('Mean absolute error is:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean squared error is:', metrics.mean_squared_error(y_test, y_pred))
print('Root mean squared error is:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


"""
-----------------------------------------------------------
📘 PROGRAM EXPLANATION
-----------------------------------------------------------
This Python program performs **Simple Linear Regression** to predict February temperatures
based on January temperature data. The process includes data loading, visualization, model
training, prediction, and evaluation.

🔹 Step-by-Step Explanation:

1️⃣ **Importing Libraries**
   - pandas, numpy → For data handling and numerical computation.
   - matplotlib, seaborn → For visualizing data and model relationships.
   - scikit-learn → For splitting data, building the linear regression model, and evaluating performance.

2️⃣ **Loading the Dataset**
   - Reads 'temperatures.csv' file into a pandas DataFrame.
   - Displays its shape (rows, columns) and statistical summary.

3️⃣ **Data Visualization**
   - Scatter plot shows the relationship between January and February temperatures.
   - Distribution plot visualizes how February temperatures are spread.

4️⃣ **Preparing Data**
   - X (input feature) = January temperatures.
   - y (target) = February temperatures.
   - Data is split into training (80%) and testing (20%) subsets.

5️⃣ **Model Training**
   - A Linear Regression model is trained using the training data.
   - The intercept and coefficient represent the best-fit line parameters.

6️⃣ **Prediction & Comparison**
   - Model predicts February temperatures using test data.
   - Actual vs Predicted values are displayed and visualized in a bar chart.

7️⃣ **Performance Evaluation**
   - Model accuracy is checked using Mean Absolute Error (MAE), 
     Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
   - Lower error values indicate better model performance.

8️⃣ **Visualization of Regression Line**
   - A red line shows the predicted trend.
   - Gray points represent the actual observed data.

-----------------------------------------------------------
⚙️ HOW TO INSTALL REQUIRED LIBRARIES
-----------------------------------------------------------
Before running this program in **Spyder (Anaconda)**, open **Anaconda Prompt** and run:

    conda install pandas matplotlib seaborn numpy scikit-learn

This ensures all dependencies for data handling, visualization, and machine learning are installed.

-----------------------------------------------------------
✅ SUMMARY
-----------------------------------------------------------
✔ Loads temperature data
✔ Visualizes the data
✔ Builds a Linear Regression model
✔ Predicts February temperatures from January
✔ Evaluates model performance using error metrics
✔ Displays comparison and regression plots

This workflow demonstrates a complete **Machine Learning pipeline** — from data preprocessing to model evaluation — using Python.
-----------------------------------------------------------
"""
