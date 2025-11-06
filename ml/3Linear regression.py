# To install required libraries in Spyder, open Anaconda Prompt and run:
# conda install pandas matplotlib seaborn numpy scikit-learn

# Importing essential libraries
import pandas as pd                # For data handling and analysis
import numpy as np                 # For numerical operations
import matplotlib.pyplot as plt    # For plotting graphs
import seaborn                     # For advanced visualization
from sklearn.model_selection import train_test_split   # For splitting dataset
from sklearn.linear_model import LinearRegression       # For linear regression model
from sklearn import metrics                             # For performance metrics

# Load the dataset
# Run this line using F9 (Run Current Line in Spyder)
dataset = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\lp-1\temperatures.csv")

# Display number of rows and columns
print(dataset.shape)

# Display statistical summary (mean, std, min, max)
print(dataset.describe())

# Plot January vs February temperature scatter plot
dataset.plot(x='JAN', y='FEB', style='o')  # 'o' means circle markers
plt.title('JAN vs FEB')                     # Title of the plot
plt.xlabel('mintemp')                       # Label for X-axis
plt.ylabel('maxtemp')                       # Label for Y-axis
plt.show()                                  # Show the plot

# Distribution plot for February temperatures
plt.figure(figsize=(15, 10))       # Set figure size
plt.tight_layout()                 # Adjust layout to prevent overlap
seaborn.distplot(dataset['FEB'])   # Plot distribution of FEB column
plt.show()

# Prepare independent (X) and dependent (y) variables
X = dataset['JAN'].values.reshape(-1, 1)   # Reshape to 2D array for sklearn
y = dataset['FEB'].values.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model parameters
print('Intercept is :', model.intercept_)   # Constant term
print('Coefficient is :', model.coef_)      # Slope (relationship strength)

# Predicting values using the test set
y_pred = model.predict(X_test)

# Create DataFrame comparing actual vs predicted values
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# Display first 25 results in bar graph
df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Plot regression line (Actual vs Predicted)
plt.scatter(X_test, y_test, color='gray')             # Actual points
plt.plot(X_test, y_pred, color='red', linewidth=2)    # Regression line
plt.show()

# Evaluate model performance using error metrics
print('Mean absolute error is:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean squared error is:', metrics.mean_squared_error(y_test, y_pred))
print('Root mean squared error is:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# End of program — Run section by section using F9 in Spyder 
#In summary, this program builds and evaluates a Linear Regression model to predict February temperatures from January data.
#It includes all key machine learning steps — loading data, visualizing relationships, training the model, testing performance, and interpreting results.
#The output helps us see how well January temperatures can explain or predict February temperatures using a linear trend.
