# lab_1_math
from math import cos, log, pi

import integrate
import numpy as np

# 1
from matplotlib import pyplot as plt

matrix = np.random.rand(5, 5)
matrix_1 = np.transpose(matrix)

print(matrix)

print(np.linalg.det(matrix))

# 2

vector = np.linalg.eig(matrix)

matrix_2 = np.random.rand(5, 5)
res = np.dot(matrix, matrix_2)
print(res)

# 3

A = np.matrix([[0, -3, -1], [3, 8, 2], [-7, -15, -3]])
vals, vecs = np.linalg.eig(A)
print("value")
print(vals)
print("vector")
print(vecs)


# 4

def f(x):
    return 1 / np.sqrt((2 * x) - 1)


v, err = integrate.quad(f, 0, 4)
print(v)

# 5

f = lambda x, y: cos(x + y)
a = lambda x: x
v, err = integrate.dblquad(f, 0, pi/2, 0, a)

# 6

x = np.linspace(-10, 10, 100)
y1 = np.log(x + 5)
y2 = 3*x - 2


fig, ax = plt.subplots()
ax.plot(x, y1, color="blue", label="y(x)")
ax.plot(x, y2, color="red", label="y'(x)")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

plt.show()
