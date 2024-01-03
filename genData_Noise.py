import numpy as np

a = 0.5
b = 0.6
c=  0.7
x= np.array([1, -1, 2, -2,3, -3,4, -4, 5, -5])
y = a * np.exp(b*x) + c  
print(y)

# Parameters for the noise
mean = 0  # Mean of the noise
std_dev = 0.1  # Standard deviation of the noise

# Generate noise with the same shape as the existing array
noise = np.random.normal(mean, std_dev, size=x.shape)
print(noise)
