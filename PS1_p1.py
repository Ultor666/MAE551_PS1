#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    pz = np.exp(x)
    nz = np.exp(-x)
    return (pz - nz) / (pz + nz)

def tanh_derivative(x):
    return 1 - tanh(x)**2

x = np.linspace(-5, 5, 1000)  # Create a range of x values
tanh_values = tanh(x)
derivative_values = tanh_derivative(x)

fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Plot tanh and its derivative
ax.plot(x, tanh_values, color="#3346FF", linewidth=3, label="tanh")
ax.plot(x, derivative_values, color="#FF335B", linewidth=3, label="derivative")

# Add legend and show plot
ax.legend(loc="upper right", frameon=True)
plt.show()


# In[ ]:




