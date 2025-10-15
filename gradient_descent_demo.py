import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**2 + 4*x + 4

def grad_f(x):
    return 2*x + 4

# Initial parameters
x = 10  # starting point
alpha = 0.1  # learning rate
steps = 20
path = [x]

# Gradient descent loop
for _ in range(steps):
    x = x - alpha * grad_f(x)
    path.append(x)

# Plot the optimization path
plt.plot(path, [f(p) for p in path], 'ro-', label="Optimization Path")
plt.title("Gradient Descent Example")
plt.xlabel("x values")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
