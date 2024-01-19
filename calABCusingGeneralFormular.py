import numpy as np
import time

# Start the timer
start_time = time.time()

def calculate_partial_derivatives(x, y, a, b, c):
    m = len(x)
    h = np.log(a *x **2 + b * x + c)
    error = (h - y)                  # J = 1/2m * np.sum((h - y)**2) 

    partial_derivative_a = np.mean((1 / m) * (error * 2 * x ** 2 / (a * x ** 2 + b * x+c)))
    partial_derivative_b = np.mean((1 / m) * (error * 2 * x  / (a * x ** 2 + b * x+c)))
    partial_derivative_c = np.mean((1 / m) * (error * 2  / (a * x ** 2 + b * x+c)))

    return partial_derivative_a, partial_derivative_b, partial_derivative_c

def gradient_descent(x, y, learning_rate, num_iterations):
    a = 1.0  # Initial guess for a
    b = 1.0  # Initial guess for b
    c = 1.0  # Initial guess for c

    for _ in range(num_iterations):
        partial_derivative_a, partial_derivative_b, partial_derivative_c = calculate_partial_derivatives(x, y, a, b, c)

        a = a - learning_rate * partial_derivative_a
        b = b - learning_rate * partial_derivative_b
        c = c - learning_rate * partial_derivative_c

    return a, b, c

# Input data
noise = np.array([-0.06461788, -0.05086664, -0.11558887 ,-0.14132715,  0.06804226,  0.01242857,
 -0.04097562 ,-0.008727 ,  -0.13992025, -0.01459187]) 
x0= np.array([1, -1, 2, -2,3, -3,4, -4, 5, -5])
y = np.array([1.79175947, 0.69314718, 2.39789527 ,1.09861229, 2.89037176, 1.79175947,
 3.29583687, 2.39789527 ,3.63758616, 2.89037176])

# Add noise to the observed y values
x = x0 + noise

# Perform gradient descent
learning_rate = 0.001
num_iterations = 1000000
a_estimated, b_estimated, c_estimated = gradient_descent(x, y, learning_rate, num_iterations)

print("Estimated values:")
print("a=", a_estimated)
print("b=", b_estimated)
print("c=", c_estimated)

end_time = time.time()
processing_time = end_time - start_time
print("Processing time:", processing_time, "seconds")
