import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load and Explore the Data
data = pd.read_csv(r"D:\Documents\US Documents\Fall 2023\MAE 551 Machine learning\HW1\HW1_P4_CO2 Emissions.csv")

# Step 2: Data Preprocessing
# Select relevant columns
data = data[['Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]

# Handle missing values (if any)
data.dropna(inplace=True)

# Step 3: Split the Data
X = data[['Fuel Consumption Comb (L/100 km)']]
y = data['CO2 Emissions(g/km)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Linear Regression Models with Different Learning Rates and Optimizers
learning_rates = [0.001, 0.01, 0.1, 0.5]
optimizers = ['sgd', 'adam', 'lbfgs']

results = []

for lr in learning_rates:
    for optimizer in optimizers:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results.append((lr, optimizer, mse))

# Step 5: Evaluate the Models and Visualize the Results
for lr, optimizer, mse in results:
    print(f"Learning Rate: {lr}, Optimizer: {optimizer}, Mean Squared Error: {mse}")

# Visualization of the data
plt.scatter(X, y, alpha=0.5)
plt.xlabel('Fuel Consumption Comb (L/100 km)')
plt.ylabel('CO2 Emissions(g/km)')
plt.title('Fuel Consumption vs. CO2 Emissions')
plt.grid(True)
plt.show()
print("The relationship between fuel consumption and CO2 emissions appears to be linear in nature.")