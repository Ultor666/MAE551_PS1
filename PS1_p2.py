import numpy as np
import sympy as sp

# Define the function f(x)
def f(x):
    return 3 * x**2 - 1

# Define the gradient of f(x)
def gradient_f(x):
    return 6 * x

# Gradient Descent function
def gradient_descent(learning_rate, num_iterations):
    # Initialize a random starting point
    x = np.random.rand()
    
    # Lists to store the history of x and f(x) values for visualization
    x_history = []
    f_history = []
    
    # Perform gradient descent
    for i in range(num_iterations):
        # Calculate the gradient of f(x) at the current point
        gradient = gradient_f(x)
        
        # Update x using gradient descent formula
        x = x - learning_rate * gradient
        
        # Append the current values to the history lists for visualization
        x_history.append(x)
        f_history.append(f(x))
    
    return x, f(x), x_history, f_history

# Set the learning rate and number of iterations
learning_rate = 0.1
num_iterations = 100

# Run gradient descent
min_x, min_f, x_history, f_history = gradient_descent(learning_rate, num_iterations)

# Print the results
print("Minimum x:", min_x)
print("Minimum f(x):", min_f)

# Visualize the convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_history, marker='o', linestyle='-', color='b')
plt.grid(True)
plt.title('Convergence of x')
plt.xlabel('Iteration')
plt.ylabel('x')

plt.subplot(1, 2, 2)
plt.plot(f_history, marker='o', linestyle='-', color='r')
plt.title('Convergence of f(x)')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.grid(True)
plt.tight_layout()
plt.show()
