import numpy as np
import time

start_time = time.time()
noise = np.array([-0.06461788, -0.05086664, -0.11558887 ,-0.14132715,  0.06804226,  0.01242857,
 -0.04097562 ,-0.008727 ,  -0.13992025, -0.01459187]) 
x0= np.array([1, -1, 2, -2,3, -3,4, -4, 5, -5])
y = np.array([ 6 , 2, 11  ,3 ,18 , 6, 27 ,11 ,38 ,18])
#J = (a * x**2 + b * x + c - y )**2

def par_derivative(a ,b ,c ,x, y):
    da = np.mean(2 * a * x ** 4 + 2 * x ** 3 * b + 2 * x**2 * c - 2 * x ** 2 * y)
    db = np.mean(2 * a * x ** 3 + 2 * x ** 2 * b + 2 * x * c - 2 * x * y)
    dc = np.mean(2 * a * x ** 2 + 2 * x  * b + 2 * c - 2 * y)
    return da, db, dc
def GD(x,y, theta, iter):
    a= 1.0
    b= 1.0
    c= 1.0
    for _ in range(iter):
        da, db, dc = par_derivative(a,b,c,x,y)
        a= a - theta * da
        b= b - theta * db
        c= c - theta * dc
    return a ,b ,c
x = x0 + noise
theta = 0.001
iter = 100000

a,b,c = GD(x,y,theta,iter)

print(a)
print(b)
print(c)

end_time = time.time()
processing_time = end_time - start_time
print("Processing time:", processing_time, "seconds")

