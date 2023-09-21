import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load and Preprocess Data
data = pd.read_csv(r"D:\Documents\US Documents\Fall 2023\MAE 551 Machine learning\HW1\HW1_P4_CO2 Emissions.csv")
data = data[['Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
data.dropna(inplace=True)

# Step 2: Data Splitting
X = data[['Fuel Consumption Comb (L/100 km)']].values
y = data['CO2 Emissions(g/km)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create a Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x

# Step 4: Choose Optimization Algorithm
learning_rate = 0.001
optimizer = optim.Adam

# Step 5: Training the Model
input_size = 1
hidden_sizes = [64, 32, 16]  # Example architecture with three hidden layers
output_size = 1

model = NeuralNetwork(input_size, hidden_sizes, output_size)
criterion = nn.MSELoss()
optimizer = optimizer(model.parameters(), lr=learning_rate)

num_epochs = 8900
for epoch in range(num_epochs):
    inputs = torch.FloatTensor(X_train)
    labels = torch.FloatTensor(y_train)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels.view(-1, 1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: Evaluate the Model
with torch.no_grad():
    test_inputs = torch.FloatTensor(X_test)
    predicted = model(test_inputs)
    mse = mean_squared_error(y_test, predicted.numpy())
    
# Convert torch tensors to numpy arrays
predicted = predicted.numpy()
print(f'Mean Squared Error on Test Data: {mse:.4f}')

# Create scatter plots for actual and predicted values with different colors
plt.scatter(y_test, predicted, c='b', label='Predicted', alpha=0.5)
plt.scatter(y_test, y_test, c='r', label='Actual', alpha=0.5)
plt.xlabel('Actual CO2 Emissions (g/km)')
plt.ylabel('Predicted CO2 Emissions (g/km)')
plt.title('Actual vs. Predicted CO2 Emissions')
plt.show()