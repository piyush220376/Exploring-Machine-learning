import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random
# Read the data
data = pd.read_csv("E:\\python_\\diabetes.csv")
print(data.describe())
print(data.info())
print(data)
print(data.columns)
print(data.corr().to_string())

# Define features and target variable
X = data[['Glucose','Pregnancies',"BloodPressure","SkinThickness","Insulin","BMI"]]
Y = data[['Outcome']]
print(X)
print(Y)

# Plotting
sns.scatterplot(x='Glucose', y='Outcome', data=data)
plt.show()

# Seed for reproducibility
random.seed(1)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Create and train the model
regr = LinearRegression()
regr.fit(X_train, Y_train)

# Print the score
print(regr.score(X_test, Y_test))
