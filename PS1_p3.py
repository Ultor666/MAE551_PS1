import numpy as np
import matplotlib.pyplot as plt

# Generate 100 equally spaced numbers between 0 and 1
x = np.linspace(0, 1, 100)

# Calculate y = sin(2πx)
y = np.sin(2 * np.pi * x)

# Add random noise to y following a Gaussian distribution
np.random.seed(0)  # Set a seed for reproducibility
noise = np.random.normal(0, 0.1, 100)  # Gaussian noise with mean 0 and standard deviation 0.1
z = y + noise

# Plot the data of x and z
plt.figure(figsize=(10, 5))
plt.scatter(x, z, label='Noisy Data', color='b', alpha=0.6)
plt.title('Noisy Data (x vs z)')
plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.grid(True)
plt.show()

# Polynomial curve fitting
def fit_polynomial(x_train, z_train, degree):
    p = np.polyfit(x_train, z_train, degree)
    return np.poly1d(p)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

# Split the data into training (80%) and testing (20%) sets
split_index = int(0.8 * len(x))
x_train, x_test = x[:split_index], x[split_index:]
z_train, z_test = z[:split_index], z[split_index:]

# Try different polynomial degrees (M) and record RMSE for training and testing data
degrees = range(1, 11)
rmse_train = []
rmse_test = []

for degree in degrees:
    model = fit_polynomial(x_train, z_train, degree)
    z_train_pred = model(x_train)
    z_test_pred = model(x_test)
    
    rmse_train.append(calculate_rmse(z_train, z_train_pred))
    rmse_test.append(calculate_rmse(z_test, z_test_pred))

# Plot RMSE vs. Degree of Polynomial
plt.figure(figsize=(10, 5))
plt.plot(degrees, rmse_train, label='Training RMSE', marker='o')
plt.plot(degrees, rmse_test, label='Testing RMSE', marker='o')
plt.title('RMSE vs. Degree of Polynomial (M)')
plt.xlabel('Degree of Polynomial (M)')
plt.ylabel('Root Mean Square Error (RMSE)')
plt.legend()
plt.grid()
plt.show()

# Find the critical M value where overfitting occurs
overfitting_threshold = 0.1
critical_degree = None

for degree, rmse_diff in enumerate(np.diff(rmse_test)):
    if rmse_diff > overfitting_threshold:
        critical_degree = degree + 1
        break

if critical_degree is not None:
    print("Overfitting occurs at degree M = {critical_degree}")
else:
    print("No clear overfitting observed in the tested range of degrees.")
